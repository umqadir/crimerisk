"""01 - Extract the public surfaces from the published model parquets.

Reads the block-group and tract candidate parquets and writes, to the tiles
work directory (never into the repo):

  work/bg_core.parquet       one row per block group, public fields only
  work/tract_core.parquet    one row per tract, public fields only
  work/county_core.parquet   population-weighted tract rollups, DISPLAY ONLY
  work/benchmarks.json       jurisdiction / state population-weighted means
  work/extract_manifest.json row counts + source parquet sha256

Nothing here is a new estimate. The county layer is a display rollup of the
published tract values and is labelled "County average" on the map.

Run:  uv run python frontend/build/01_extract.py
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from crschema import (  # noqa: E402
    BG_SRC,
    COMPOSITES,
    DIRECT_SOURCE_MODES,
    FIPS_TO_USPS,
    OFFENSES,
    POPULATION_COL,
    SHORT,
    COUNTY_MIN_POPULATION,
    COUNTY_MIN_TRACTS,
    SUPPORT_MIN_BLOCK_GROUPS,
    SUPPORT_MIN_POPULATION,
    SPECIAL_USE_CODE,
    repo_path,
    TRACT_ONLY_OFFENSES,
    TRACT_SRC,
    WORK,
    YEAR,
)

JURIS_MASTER = repo_path("state/reference/jurisdiction_master.parquet")

BG_ID = "block_group_geoid"
TRACT_ID = "tract_id"

OFFENSE_DETAIL = [
    "index_{o}_primary",
    "index_{o}_resident",
    "rate_{o}_primary",
    "rate_{o}_resident",
    "expected_count_{o}",
    "primary_denominator_{o}",
    "index_{o}_primary_p10",
    "index_{o}_primary_p90",
    "reliability_tier_{o}",
    "source_mode_{o}",
]

COMPOSITE_COLS = [c for k in COMPOSITES for c in COMPOSITES[k][:2]]


def offense_cols() -> list[str]:
    return [tpl.format(o=o) for o in OFFENSES for tpl in OFFENSE_DETAIL]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_columns() -> dict[str, str]:
    """Public measure key -> source column name (20 map measures)."""
    out: dict[str, str] = {}
    for k, (expo, resi, _label) in COMPOSITES.items():
        out[f"x_{k}"] = expo
        out[f"r_{k}"] = resi
    for o in OFFENSES:
        out[f"x_{SHORT[o]}"] = f"index_{o}_primary"
        out[f"r_{SHORT[o]}"] = f"index_{o}_resident"
    return out


MEASURE_COLS = measure_columns()
RARE_KEYS = {f"{p}_{SHORT[o]}" for o in TRACT_ONLY_OFFENSES for p in ("x", "r")}


def source_mask(df: pd.DataFrame) -> pd.Series:
    """Seven-bit mask: bit i set when offence i is not from direct records."""
    mask = np.zeros(len(df), dtype="int16")
    for i, o in enumerate(OFFENSES):
        col = df[f"source_mode_{o}"].astype("string")
        modeled = ~col.isin(DIRECT_SOURCE_MODES)
        mask |= (modeled.to_numpy() * (1 << i)).astype("int16")
    return pd.Series(mask, index=df.index)


def load(src, id_col: str, juris_col: str) -> pd.DataFrame:
    cols = [
        id_col,
        "state_fips",
        POPULATION_COL,
        "land_area_sq_mi",
        "special_use_type",
        juris_col,
    ] + COMPOSITE_COLS + offense_cols()
    if id_col == BG_ID:
        cols.insert(1, TRACT_ID)
    df = pd.read_parquet(src, columns=cols)
    df[id_col] = df[id_col].astype(str)
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["st"] = df["state_fips"].map(FIPS_TO_USPS)
    df = df.rename(columns={juris_col: "juris_id"})
    df["juris_id"] = df["juris_id"].astype("string")
    unknown = sorted(set(df["special_use_type"].dropna().unique()) - set(SPECIAL_USE_CODE))
    if unknown:
        raise ValueError(f"Unknown special_use_type values: {unknown}")
    df["su"] = df["special_use_type"].map(SPECIAL_USE_CODE).fillna(0).astype("int16")
    df["src"] = source_mask(df)
    for key, col in MEASURE_COLS.items():
        df[key] = pd.to_numeric(df[col], errors="coerce").round(1)
    df["pop"] = pd.to_numeric(df[POPULATION_COL], errors="coerce").round().astype("Int64")
    return df


def index_definition(tr: pd.DataFrame) -> dict:
    """Recover the exact closed form of every published index from the surface.

    The published values satisfy, to floating-point equality:

        rate_{o}_primary  = 1e5 * expected_count_o / primary_denominator_o
        index_{o}_primary = 100 * rate_{o}_primary / A_o        (A_o constant)
        rate_{o}_resident = 1e5 * expected_count_o / population
        index_{o}_resident= 100 * rate_{o}_resident / B_o       (B_o constant)

    and each composite is a fixed-weight linear combination of the seven
    offence indexes, the weights summing to one. Recovering A_o, B_o and the
    weights here (instead of hard-coding them) means a county rollup built from
    summed counts lands on exactly the published scale, and a change in the
    model upstream fails this step loudly instead of silently rescaling the
    county layer.
    """
    out = {"anchor_primary": {}, "anchor_resident": {}, "weights": {}}
    for o in OFFENSES:
        for kind, key in (("primary", "anchor_primary"), ("resident", "anchor_resident")):
            rate = pd.to_numeric(tr[f"rate_{o}_{kind}"], errors="coerce")
            idx = pd.to_numeric(tr[f"index_{o}_{kind}"], errors="coerce")
            ok = rate.gt(0) & idx.gt(0)
            a = (rate[ok] / idx[ok] * 100.0)
            lo, hi = float(a.min()), float(a.max())
            if not np.isfinite(lo) or hi - lo > 1e-6 * max(abs(hi), 1.0):
                raise ValueError(
                    f"{o} {kind} index is not a constant-anchor rate "
                    f"(anchor spans {lo!r}..{hi!r}); the county rollup contract is broken"
                )
            out[key][o] = float(a.median())

    X = np.column_stack(
        [pd.to_numeric(tr[f"index_{o}_primary"], errors="coerce").to_numpy() for o in OFFENSES]
    )
    Xr = np.column_stack(
        [pd.to_numeric(tr[f"index_{o}_resident"], errors="coerce").to_numpy() for o in OFFENSES]
    )
    for key, (expo, resi, _label) in COMPOSITES.items():
        for col, design, slot in ((expo, X, "exposure"), (resi, Xr, "resident")):
            y = pd.to_numeric(tr[col], errors="coerce").to_numpy()
            m = np.isfinite(y) & np.isfinite(design).all(axis=1)
            w, *_ = np.linalg.lstsq(design[m], y[m], rcond=None)
            w = np.where(np.abs(w) < 1e-9, 0.0, w)
            err = float(np.abs(design[m] @ w - y[m]).max())
            if err > 1e-6 or abs(w.sum() - 1.0) > 1e-6:
                raise ValueError(
                    f"composite {col} is not a fixed-weight mean of the offence "
                    f"indexes (max error {err:.3e}, weights sum {w.sum():.6f})"
                )
            out["weights"][f"{key}_{slot}"] = [float(v) for v in w]
    return out


def count_weighted(df: pd.DataFrame, group: str, defn: dict) -> pd.DataFrame:
    """The 20 map measures for an aggregate, from summed counts over summed
    denominators.

    This is the only aggregation the product uses. A mean of neighborhood
    indexes gives every polygon one vote, so a 52-resident cell can set the
    colour of a county or the benchmark of a small city. Summing the counts and
    the denominators first, then indexing on the anchors recovered from the
    published surface, gives each cell exactly the weight its own denominator
    earns and lands on the published scale.
    """
    g = df[group]
    pop = pd.to_numeric(df["pop"], errors="coerce").astype("float64").fillna(0.0)
    index = sorted(x for x in g.dropna().unique())
    out = pd.DataFrame(index=index)
    out.index.name = group
    pop_sum = pop.groupby(g).sum().reindex(index).fillna(0.0)

    per_offense = {}
    for o in OFFENSES:
        ec = pd.to_numeric(df[f"expected_count_{o}"], errors="coerce").fillna(0.0)
        den = pd.to_numeric(df[f"primary_denominator_{o}"], errors="coerce").fillna(0.0)
        ec_sum = ec.groupby(g).sum().reindex(index)
        den_sum = den.groupby(g).sum().reindex(index)
        per_offense[("x", o)] = (
            100.0 * (1e5 * ec_sum / den_sum.replace(0.0, np.nan)) / defn["anchor_primary"][o]
        )
        per_offense[("r", o)] = (
            100.0 * (1e5 * ec_sum / pop_sum.replace(0.0, np.nan)) / defn["anchor_resident"][o]
        )

    for o in OFFENSES:
        out[f"x_{SHORT[o]}"] = per_offense[("x", o)]
        out[f"r_{SHORT[o]}"] = per_offense[("r", o)]

    for key in COMPOSITES:
        for prefix, slot in (("x", "exposure"), ("r", "resident")):
            w = defn["weights"][f"{key}_{slot}"]
            total = None
            for i, o in enumerate(OFFENSES):
                if w[i] == 0.0:
                    continue
                part = per_offense[(prefix, o)] * w[i]
                total = part if total is None else total + part
            out[f"{prefix}_{key}"] = total

    for key in MEASURE_COLS:
        out[key] = out[key].round(1)
    out["pop"] = pop_sum.round().astype("int64")
    return out


def county_rollup(tr: pd.DataFrame, defn: dict) -> pd.DataFrame:
    """County rates plus the support floor. Counties below it publish nothing."""
    out = count_weighted(tr, "county_geoid", defn)
    n_tracts = tr["county_geoid"].value_counts().reindex(out.index).fillna(0).astype("int64")
    out["n_tracts"] = n_tracts
    supported = (n_tracts >= COUNTY_MIN_TRACTS) & (out["pop"] >= COUNTY_MIN_POPULATION)
    out["supported"] = supported.astype("int16")
    for key in MEASURE_COLS:
        out.loc[~supported, key] = np.nan
    return out


def weighted_means(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Population-weighted mean of every map measure, grouped by `by`.

    Weights are resident population; cells with a null measure are excluded
    from that measure's weight, so a measure's mean is over the population it
    is actually published for.
    """
    w = pd.to_numeric(df["pop"], errors="coerce").astype("float64").fillna(0.0)
    out = {}
    g = df[by]
    for key in MEASURE_COLS:
        v = pd.to_numeric(df[key], errors="coerce").astype("float64")
        ok = v.notna() & (w > 0)
        num = pd.Series(np.where(ok, v * w, 0.0), index=df.index).groupby(g).sum()
        den = pd.Series(np.where(ok, w, 0.0), index=df.index).groupby(g).sum()
        out[key] = (num / den.replace(0.0, np.nan)).round(1)
    res = pd.DataFrame(out)
    res["pop"] = w.groupby(g).sum().round().astype("int64")
    return res


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"Reading block groups: {BG_SRC.name}")
    bg = load(BG_SRC, BG_ID, "eb_jurisdiction_id")
    bg[TRACT_ID] = bg[TRACT_ID].astype(str).str.zfill(11)
    print(f"  {len(bg):,} block groups, {bg['st'].nunique()} states")

    print(f"Reading tracts: {TRACT_SRC.name}")
    tr = load(TRACT_SRC, TRACT_ID, "dominant_eb_jurisdiction_id")
    tr[TRACT_ID] = tr[TRACT_ID].astype(str).str.zfill(11)
    print(f"  {len(tr):,} tracts")

    # Block groups carry NULL murder/rape by policy; the map routes those two
    # offences to the tract layer, so nothing is back-filled onto block groups.
    for key in sorted(RARE_KEYS):
        n = int(bg[key].notna().sum())
        if n:
            raise ValueError(f"block-group {key} unexpectedly non-null in {n:,} rows")

    # --- county rollups (display only) -------------------------------------
    print("Recovering the published index definition from the tract surface")
    defn = index_definition(tr)
    print("Rolling tracts up to counties (count-weighted rates, display only)")
    tr["county_geoid"] = tr[TRACT_ID].str[:5]
    prev = weighted_means(tr, "county_geoid")     # for the before/after report
    county = county_rollup(tr, defn)
    county = county.reset_index()
    county["st"] = county["county_geoid"].str[:2].map(FIPS_TO_USPS)
    # County source flag: bit set when the majority of county residents live in
    # cells whose value for that offence is modeled rather than direct.
    w = pd.to_numeric(tr["pop"], errors="coerce").astype("float64").fillna(0.0)
    mask = np.zeros(len(county), dtype="int16")
    order = {g: i for i, g in enumerate(county["county_geoid"])}
    idx = tr["county_geoid"].map(order).to_numpy()
    tot = np.zeros(len(county))
    np.add.at(tot, idx, w.to_numpy())
    for i, o in enumerate(OFFENSES):
        modeled = (~tr[f"source_mode_{o}"].astype("string").isin(DIRECT_SOURCE_MODES)).to_numpy()
        part = np.zeros(len(county))
        np.add.at(part, idx, w.to_numpy() * modeled)
        mask |= ((part > 0.5 * np.maximum(tot, 1e-9)) * (1 << i)).astype("int16")
    county["src"] = mask
    county["su"] = 0
    n_floor = int((county["supported"] == 0).sum())
    print(
        f"  {len(county):,} counties; {n_floor:,} below the support floor "
        f"(<{COUNTY_MIN_TRACTS} tracts or <{COUNTY_MIN_POPULATION:,} residents)"
    )
    shift = (
        prev["r_ov"].reindex(county["county_geoid"]).to_numpy()
        - county["r_ov"].to_numpy()
    )
    finite = np.isfinite(shift)
    print(
        f"  per-resident overall index vs the old population-weighted mean: "
        f"median shift {np.nanmedian(np.abs(shift[finite])):.1f}, "
        f"max {np.nanmax(np.abs(shift[finite])):.1f} over {int(finite.sum()):,} counties"
    )

    # --- benchmarks ---------------------------------------------------------
    # Same count-weighted rate as the county rollup: a 300-resident village's
    # benchmark must not be one block group's index.
    print("Computing jurisdiction and state count-weighted rates")
    juris_bg = count_weighted(bg, "juris_id", defn)
    juris_tr = count_weighted(tr, "juris_id", defn)
    # Murder and rape only exist at tract support, so their jurisdiction and
    # state benchmarks come from the tract surface; everything else from the
    # finer block-group surface, whose jurisdiction assignment is exact.
    juris = juris_bg.copy()
    for key in sorted(RARE_KEYS):
        juris[key] = juris_tr[key].reindex(juris.index)
    juris["pop"] = juris_bg["pop"]
    juris_cells = bg.groupby("juris_id").size().reindex(juris.index).fillna(0)
    juris_supported = (
        (juris_cells >= SUPPORT_MIN_BLOCK_GROUPS) & (juris["pop"] >= SUPPORT_MIN_POPULATION)
    )
    juris["n_block_groups"] = juris_cells.astype("int64")
    juris["supported"] = juris_supported.astype("int16")
    print(
        f"  {int((~juris_supported).sum()):,} of {len(juris):,} jurisdictions below the "
        f"support floor (<{SUPPORT_MIN_BLOCK_GROUPS} block groups or "
        f"<{SUPPORT_MIN_POPULATION:,} residents); they cover "
        f"{100 * juris.loc[~juris_supported, 'pop'].sum() / juris['pop'].sum():.2f}% "
        "of residents and their term is dropped from the card"
    )

    state_bg = count_weighted(bg, "st", defn)
    state_tr = count_weighted(tr, "st", defn)
    state = state_bg.copy()
    for key in sorted(RARE_KEYS):
        state[key] = state_tr[key].reindex(state.index)

    jm = pd.read_parquet(JURIS_MASTER, columns=["jurisdiction_id", "jurisdiction_name", "state_abbr"])
    name_by_id = dict(zip(jm["jurisdiction_id"].astype(str), jm["jurisdiction_name"].astype(str)))

    def juris_display(jid: str) -> str:
        if jid in name_by_id:
            return name_by_id[jid]
        # Statewide non-municipal remainder ids look like "01:state_nonmunicipal_remainder".
        parts = jid.split(":")
        if len(parts) >= 2 and parts[1] == "state_nonmunicipal_remainder":
            from crschema import STATE_NAME

            abbr = FIPS_TO_USPS.get(parts[0], "")
            return f"Unincorporated {STATE_NAME.get(abbr, abbr)}"
        return jid

    benchmarks = {
        "measures": sorted(MEASURE_COLS),
        "us": {k: 100.0 for k in MEASURE_COLS},
        "state": {
            str(k): {m: (None if pd.isna(v[m]) else float(v[m])) for m in MEASURE_COLS}
            for k, v in state.iterrows()
        },
        "jurisdiction": {
            str(k): {
                "name": juris_display(str(k)),
                "supported": int(v["supported"]),
                **{m: (None if pd.isna(v[m]) else float(v[m])) for m in MEASURE_COLS},
            }
            for k, v in juris.iterrows()
        },
    }
    (WORK / "benchmarks.json").write_text(json.dumps(benchmarks))
    print(
        f"  {len(benchmarks['jurisdiction']):,} jurisdictions, "
        f"{len(benchmarks['state'])} states"
    )

    bg.to_parquet(WORK / "bg_core.parquet", index=False)
    tr.to_parquet(WORK / "tract_core.parquet", index=False)
    county.to_parquet(WORK / "county_core.parquet", index=False)

    manifest = {
        "year": YEAR,
        "n_block_groups": int(len(bg)),
        "n_tracts": int(len(tr)),
        "n_counties": int(len(county)),
        "n_counties_below_support_floor": int((county["supported"] == 0).sum()),
        "county_support_floor": {
            "min_tracts": COUNTY_MIN_TRACTS,
            "min_population": COUNTY_MIN_POPULATION,
        },
        "index_definition": defn,
        "states": sorted(bg["st"].dropna().unique().tolist()),
        "source": {
            "block_group_parquet": str(BG_SRC),
            "block_group_sha256": sha256(BG_SRC),
            "block_group_bytes": BG_SRC.stat().st_size,
            "tract_parquet": str(TRACT_SRC),
            "tract_sha256": sha256(TRACT_SRC),
            "tract_bytes": TRACT_SRC.stat().st_size,
        },
    }
    (WORK / "extract_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {WORK}/bg_core.parquet, tract_core.parquet, county_core.parquet")


if __name__ == "__main__":
    main()
