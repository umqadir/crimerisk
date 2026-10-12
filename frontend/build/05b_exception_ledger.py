"""05b - Public exception ledger: large Overall-exposure jumps across jurisdiction lines.

One row per pair of rook-adjacent block groups that
  * both publish an Overall crime exposure index,
  * sit in different police jurisdictions (`eb_jurisdiction_id`), and
  * differ by a factor of at least RATIO_THRESHOLD (higher / lower index).

Each side carries its jurisdiction id and name, the distinct `level_repair_mode`
values across the seven offenses (how the jurisdiction total was obtained;
jurisdiction_controls_<year>.parquet), and the source years of the smoothed
jurisdiction total: the earliest `latest_clean_year` and the smallest
`clean_year_count` across the seven offenses
(jurisdiction_controls_smoothed_<year>.parquet). Both are the files the candidate
run's manifest records as its inputs, checked by size. The block-group parquet's
own level_repair_mode_<offense> is null on 30% of block groups (the statewide
remainders), so the jurisdiction table is the source.

The threshold is fixed before looking at the result and is not a parameter.

  dist/site/downloads/crimerisk_<year>_exception_ledger.csv

Run:  uv run python frontend/build/05b_exception_ledger.py [--candidate DIR]
      (default candidate: the directory of the frozen snapshot parquet)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import (  # noqa: E402
    BG_SRC,
    COMPOSITES,
    DIST,
    FIPS_TO_USPS,
    STATE_NAME,
    YEAR,
    repo_path,
)

RATIO_THRESHOLD = 4.0
OVERALL_EXPOSURE = COMPOSITES["ov"][0]
EDGES = repo_path("state/geometry/bg_rook_edges_2020.parquet")
JURIS_MASTER = repo_path("state/reference/jurisdiction_master.parquet")
BG_ID = "block_group_geoid"
JURIS = "eb_jurisdiction_id"

SIDE_FIELDS = [
    "block_group",
    "overall_exposure",
    "jurisdiction_id",
    "jurisdiction_name",
    "level_repair_mode",
    "total_latest_clean_year",
    "total_clean_years",
]
COLUMNS = (
    [f"{f}_a" for f in SIDE_FIELDS[:2]]
    + [f"{f}_b" for f in SIDE_FIELDS[:2]]
    + ["ratio"]
    + [f"{f}_a" for f in SIDE_FIELDS[2:]]
    + [f"{f}_b" for f in SIDE_FIELDS[2:]]
)


def juris_display(jid: str, names: dict[str, str]) -> str:
    if jid in names:
        return names[jid]
    parts = jid.split(":")
    if len(parts) >= 2 and parts[1] == "state_nonmunicipal_remainder":
        abbr = FIPS_TO_USPS.get(parts[0], "")
        return f"Unincorporated {STATE_NAME.get(abbr, abbr)}"
    return jid


def jurisdiction_totals(controls: pd.DataFrame, smoothed: pd.DataFrame) -> pd.DataFrame:
    """Per jurisdiction: distinct repair modes (';'-joined), the earliest
    latest_clean_year and the fewest clean years, each over the seven offenses."""
    modes = (
        controls.dropna(subset=["level_repair_mode"])
        .groupby("jurisdiction_id")["level_repair_mode"]
        .agg(lambda v: ";".join(sorted(set(v.astype(str)))))
    )
    g = smoothed.groupby("jurisdiction_id")
    out = pd.DataFrame({
        "level_repair_mode": modes,
        "total_latest_clean_year": g["latest_clean_year"].min(),
        "total_clean_years": g["clean_year_count"].min(),
    })
    out.index = out.index.astype(str)
    return out


def ledger(
    bg: pd.DataFrame,
    edges: pd.DataFrame,
    names: dict[str, str],
    totals: pd.DataFrame,
) -> pd.DataFrame:
    """The exception rows, highest ratio first.

    bg      block_group_geoid, eb_jurisdiction_id and the Overall exposure field
    edges   left_geoid, right_geoid (either orientation, duplicates allowed)
    totals  jurisdiction_totals(), indexed by jurisdiction id
    """
    frame = bg.copy()
    frame[BG_ID] = frame[BG_ID].astype(str)
    frame["value"] = pd.to_numeric(frame[OVERALL_EXPOSURE], errors="coerce")
    frame["juris"] = frame[JURIS].astype("string")
    frame = frame.set_index(BG_ID)

    e = edges[["left_geoid", "right_geoid"]].astype(str)
    lo = np.minimum(e["left_geoid"].to_numpy(), e["right_geoid"].to_numpy())
    hi = np.maximum(e["left_geoid"].to_numpy(), e["right_geoid"].to_numpy())
    pairs = pd.DataFrame({"g1": lo, "g2": hi}).drop_duplicates()
    pairs = pairs[pairs["g1"] != pairs["g2"]]
    pairs = pairs[pairs["g1"].isin(frame.index) & pairs["g2"].isin(frame.index)]

    v1 = frame["value"].reindex(pairs["g1"]).to_numpy()
    v2 = frame["value"].reindex(pairs["g2"]).to_numpy()
    j1 = frame["juris"].reindex(pairs["g1"]).to_numpy()
    j2 = frame["juris"].reindex(pairs["g2"]).to_numpy()
    published = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
    known = pd.notna(j1) & pd.notna(j2)
    differ = known & (j1.astype(object) != j2.astype(object))
    keep = published & differ
    v1, v2 = v1[keep], v2[keep]
    g1, g2 = pairs["g1"].to_numpy()[keep], pairs["g2"].to_numpy()[keep]
    ratio = np.maximum(v1, v2) / np.minimum(v1, v2)
    big = ratio >= RATIO_THRESHOLD
    first_is_high = v1 >= v2
    a = np.where(first_is_high, g1, g2)[big]
    b = np.where(first_is_high, g2, g1)[big]
    ratio = ratio[big]

    def side(ids: np.ndarray, suffix: str) -> pd.DataFrame:
        rows = frame.loc[ids]
        juris = rows["juris"].astype(str).to_numpy()
        tot = totals.reindex(juris)
        return pd.DataFrame({
            f"block_group_{suffix}": ids,
            f"overall_exposure_{suffix}": rows["value"].round(1).to_numpy(),
            f"jurisdiction_id_{suffix}": juris,
            f"jurisdiction_name_{suffix}": [juris_display(j, names) for j in juris],
            f"level_repair_mode_{suffix}": tot["level_repair_mode"].to_numpy(),
            f"total_latest_clean_year_{suffix}": tot["total_latest_clean_year"].astype("Int64").array,
            f"total_clean_years_{suffix}": tot["total_clean_years"].astype("Int64").array,
        })

    out = pd.concat([side(a, "a"), side(b, "b")], axis=1)
    out["ratio"] = np.round(ratio, 2)
    out = out[COLUMNS].sort_values(
        ["ratio", "block_group_a", "block_group_b"], ascending=[False, True, True]
    )
    return out.reset_index(drop=True)


def candidate_input(candidate: Path, key: str) -> Path:
    """An input file the candidate run read (manifest input_file_stats), checked by size."""
    man = json.loads((candidate / "manifest.json").read_text())
    rec = man["input_file_stats"][key]
    path = Path(rec["path"])
    local = repo_path("state/" + path.as_posix().split("/state/", 1)[1])
    if not local.exists() or local.stat().st_size != int(rec["size_bytes"]):
        raise SystemExit(
            f"{local} does not match the candidate manifest ({rec['size_bytes']} bytes); "
            "it would not describe this candidate"
        )
    return local


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, default=BG_SRC.parent)
    args = ap.parse_args()
    out_path = DIST / "site" / "downloads" / f"crimerisk_{YEAR}_exception_ledger.csv"

    src = args.candidate / f"crimerisk_block_group_{YEAR}_ags_core.parquet"
    cols = [BG_ID, JURIS, OVERALL_EXPOSURE]
    print(f"Reading {src}")
    bg = pd.read_parquet(src, columns=cols)
    edges = pd.read_parquet(EDGES, columns=["left_geoid", "right_geoid"])
    jm = pd.read_parquet(JURIS_MASTER, columns=["jurisdiction_id", "jurisdiction_name"])
    names = dict(zip(jm["jurisdiction_id"].astype(str), jm["jurisdiction_name"].astype(str)))
    controls = pd.read_parquet(
        candidate_input(args.candidate, "jurisdiction_controls"),
        columns=["jurisdiction_id", "offense", "level_repair_mode"],
    )
    smoothed = pd.read_parquet(
        candidate_input(args.candidate, "jurisdiction_controls_smoothed"),
        columns=["jurisdiction_id", "offense", "latest_clean_year", "clean_year_count"],
    )

    out = ledger(bg, edges, names, jurisdiction_totals(controls, smoothed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    juris_pairs = out[["jurisdiction_id_a", "jurisdiction_id_b"]].apply(
        lambda r: tuple(sorted(r)), axis=1
    ).nunique()
    print(
        f"{len(out):,} adjacent cross-jurisdiction block-group pairs at ratio >= "
        f"{RATIO_THRESHOLD:g} ({juris_pairs:,} jurisdiction pairs); wrote {out_path}"
    )


if __name__ == "__main__":
    main()
