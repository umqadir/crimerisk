"""04 - GEOID-keyed lookup shards and per-state benchmark files.

Everything the selection card needs that is NOT painted on the map lives here,
so the tiles stay at 25 attributes and a click costs one small cached fetch.
No hidden map instances, no second tile read.

  dist/site/data/shards/{bg,tract,county}/<prefix>.json
  dist/site/data/shards/{bg,tract,county}/index.json   prefixes that were split
  dist/site/data/bench/<ST>.json                        jurisdiction + state means

Shard record (array, positional - see FIELDS below):
  [name, county_name, jurisdiction_id, population, land_area_sq_mi,
   special_use_code, [measure value x20], [expected_count x7],
   [source_mode_code x7],
   [direct expected-count share x3: overall, violent, property],
   [level total reported x7: 1 reported, 0 estimated, null unknown]]

The last array answers a question the source mode does not: whether the AGENCY
TOTAL this neighborhood's share was cut from was filed for the data year, or
reconstructed from its own history, peer agencies, a partial year or a benchmark.
A cell can be modeled from a reported total, or allocated from an estimated one,
and the card says both.

2025.1 withholds the p10/p90 index range and the reliability tier, so neither
appears in the record. Positions after `expected_count` moved accordingly.

The direct share is the fraction of a composite's expected count that comes
from offences whose source_mode is direct_city_incident. The card picks its
source phrase from that number rather than from an all-offences-direct test, so
one modeled rare offence cannot relabel a composite that is almost entirely
built from incident records.

The 20 measure values are repeated here (they are also in the tiles) so a card,
a shared URL or a compare view can be rendered without the feature being on
screen. Measure order is the `measures` list in every bench file.

Offence order is crschema.OFFENSES. Composite counts and the composite
source phrase are derived in the client from these seven.

Run:  uv run python frontend/build/04_shards.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import (  # noqa: E402
    BG_SRC,
    DIST,
    LEVEL_REPORTED_REPAIR_MODES,
    LEVEL_REPORTED_STATUS,
    MEASURE_ORDER,
    OFFENSES,
    SOURCE_MODE_CODE,
    STATE_NAME,
    TRACT_SRC,
    WORK,
)

SITE = DIST / "site"
SHARD_DIR = SITE / "data" / "shards"
BENCH_DIR = SITE / "data" / "bench"
MAX_SHARD_BYTES = 64 * 1024
BASE_DEPTH = {"bg": 5, "tract": 5, "county": 2}

FIELDS = [
    "name",
    "county",
    "jurisdiction_id",
    "population",
    "land_area_sq_mi",
    "special_use_code",
    "measure_values",
    "expected_count",
    "source_mode_code",
    "direct_share",
    "level_total_reported",
]

# Composite membership, by index into OFFENSES.
COMPOSITE_MEMBERS = {
    "ov": list(range(len(OFFENSES))),
    "vi": [OFFENSES.index(o) for o in ["murder", "rape", "robbery", "aggravated_assault"]],
    "pr": [OFFENSES.index(o) for o in ["burglary", "larceny", "motor_vehicle_theft"]],
}
COMPOSITE_ORDER = ["ov", "vi", "pr"]

# A `mixed` cell is counted as half direct. Mixed is vanishingly rare (63 of
# 238,193 block groups on the largest offence) and the exact split is not
# carried into the public surface.
MODE_DIRECT_WEIGHT = {0: 1.0, 1: 0.5, 2: 0.0}

# Census place/county legal suffixes stripped for display.
SUFFIX_RE = re.compile(
    r"\s+(city|town|village|borough|township|municipality|CDP|plantation|gore|"
    r"grant|location|purchase|charter township|urban county|metro government|"
    r"consolidated government|unified government)$",
    re.IGNORECASE,
)


def display_name(raw: str) -> str:
    out = SUFFIX_RE.sub("", raw).strip()
    return out or raw


def num(x, dp: int):
    if x is None or (isinstance(x, float) and x != x) or x is pd.NA:
        return None
    v = round(float(x), dp)
    return int(v) if dp == 0 else v


def direct_shares(ec: dict, sm: dict, n: int) -> dict:
    """Direct expected-count share per composite, as float arrays of length n."""
    out = {}
    for key in COMPOSITE_ORDER:
        num = np.zeros(n)
        den = np.zeros(n)
        for i in COMPOSITE_MEMBERS[key]:
            o = OFFENSES[i]
            c = np.nan_to_num(ec[o], nan=0.0)
            w = np.vectorize(lambda m: MODE_DIRECT_WEIGHT.get(int(m), 0.0))(sm[o])
            num += c * w
            den += c
        out[key] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


def level_total_reported(src, id_col: str) -> dict:
    """Per-offense two-state flag: was the jurisdiction total filed for the data year?

    1  the agency filed a complete year and it was admitted as observed.
    0  the total was repaired: decayed own history, a pooled peer unit, an
       annualized partial, or benchmark mass. The card calls all of these
       "estimated, not reported in full", because from a reader's point of view
       they are the same fact -- nobody filed this year's complete number.
    None  no jurisdiction total applies to the cell at all, which is the statewide
       non-municipal remainder.

    Read straight from the published parquet rather than from `bg_core`, so the
    flag cannot drift from the surface it describes.
    """
    columns = [id_col] + [
        f"level_{part}_{offense}"
        for offense in OFFENSES
        for part in ("admission_status", "repair_mode")
    ]
    frame = pd.read_parquet(src, columns=columns)
    ids = frame[id_col].astype(str).to_numpy()
    flags = []
    for offense in OFFENSES:
        status = frame[f"level_admission_status_{offense}"].astype("string")
        repair = frame[f"level_repair_mode_{offense}"].astype("string")
        known = status.notna()
        reported = status.eq(LEVEL_REPORTED_STATUS) & repair.fillna("none").isin(
            LEVEL_REPORTED_REPAIR_MODES
        )
        value = pd.Series(pd.NA, index=frame.index, dtype="Int8")
        value[known] = reported[known].astype("int8")
        flags.append(value.to_numpy(dtype=object, na_value=None))
    return {
        ids[i]: [None if flags[j][i] is None else int(flags[j][i]) for j in range(len(OFFENSES))]
        for i in range(len(frame))
    }


def build_records(
    core: pd.DataFrame,
    id_col: str,
    names: dict,
    county_names: dict,
    tract_names: dict | None,
    level_reported: dict,
) -> dict:
    vals = {m: core[m].to_numpy(dtype="float64") for m in MEASURE_ORDER}
    ec = {o: core[f"expected_count_{o}"].to_numpy(dtype="float64") for o in OFFENSES}
    sm = {
        o: core[f"source_mode_{o}"].astype("string").map(SOURCE_MODE_CODE).fillna(2).to_numpy(dtype="int64")
        for o in OFFENSES
    }
    ids = core[id_col].to_numpy()
    juris = core["juris_id"].astype("string").fillna("").to_numpy()
    pop = pd.to_numeric(core["pop"], errors="coerce").to_numpy(dtype="float64")
    area = core["land_area_sq_mi"].to_numpy(dtype="float64")
    su = core["su"].to_numpy(dtype="int64")

    ds = direct_shares(ec, sm, len(core))

    out: dict[str, list] = {}
    for i in range(len(core)):
        gid = str(ids[i])
        nm = names.get(gid, "")
        if tract_names is not None:
            tn = tract_names.get(gid[:11], "")
            nm = f"{nm}, {tn}" if tn else nm
        out[gid] = [
            nm,
            county_names.get(gid[:5], ""),
            juris[i],
            num(pop[i], 0),
            num(area[i], 3),
            int(su[i]),
            [num(vals[m][i], 1) for m in MEASURE_ORDER],
            [num(ec[o][i], 2) for o in OFFENSES],
            [int(sm[o][i]) for o in OFFENSES],
            [num(ds[k][i], 3) for k in COMPOSITE_ORDER],
            level_reported.get(gid),
        ]
    return out


def shard(records: dict, geo: str) -> dict:
    """Prefix-shard to <= MAX_SHARD_BYTES, splitting deeper where needed."""
    out_dir = SHARD_DIR / geo
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    split_prefixes: set[str] = set()
    written = 0
    max_bytes = 0

    def emit(keys: list[str], depth: int) -> None:
        nonlocal written, max_bytes
        buckets: dict[str, list[str]] = {}
        for k in keys:
            buckets.setdefault(k[:depth], []).append(k)
        for prefix, ks in buckets.items():
            payload = json.dumps({k: records[k] for k in sorted(ks)}, separators=(",", ":"))
            data = payload.encode()
            if len(data) > MAX_SHARD_BYTES and depth < len(ks[0]):
                split_prefixes.add(prefix)
                emit(ks, depth + 1)
                continue
            (out_dir / f"{prefix}.json").write_bytes(data)
            written += 1
            max_bytes = max(max_bytes, len(data))

    emit(sorted(records), BASE_DEPTH[geo])
    (out_dir / "index.json").write_text(
        json.dumps({"base_depth": BASE_DEPTH[geo], "split": sorted(split_prefixes)}, separators=(",", ":"))
    )
    total = sum(p.stat().st_size for p in out_dir.glob("*.json"))
    return {
        "records": len(records),
        "shards": written,
        "split_prefixes": len(split_prefixes),
        "max_shard_bytes": max_bytes,
        "total_bytes": total,
        "index_bytes": (out_dir / "index.json").stat().st_size,
    }


def main() -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    names_county = pd.read_parquet(WORK / "names_county.parquet")
    county_names = dict(zip(names_county["geoid"].astype(str), names_county["name"].astype(str)))
    names_tract = pd.read_parquet(WORK / "names_tract.parquet")
    tract_names = dict(zip(names_tract["geoid"].astype(str), names_tract["name"].astype(str)))
    names_bg = pd.read_parquet(WORK / "names_bg.parquet")
    bg_names = dict(zip(names_bg["geoid"].astype(str), names_bg["name"].astype(str)))

    stats = {}

    print("County shards")
    county = pd.read_parquet(WORK / "county_core.parquet")
    tr_for_county = pd.read_parquet(
        WORK / "tract_core.parquet",
        columns=["tract_id"]
        + [f"expected_count_{o}" for o in OFFENSES]
        + [f"source_mode_{o}" for o in OFFENSES],
    )
    tr_for_county["county_geoid"] = tr_for_county["tract_id"].astype(str).str.zfill(11).str[:5]
    # A county's source phrase is the same expected-count share, summed over the
    # tracts it contains, so the rollup card says the same kind of thing as the
    # neighborhood card underneath it.
    county_ds: dict[str, dict[str, float]] = {}
    for key in COMPOSITE_ORDER:
        direct_mass = pd.Series(0.0, index=tr_for_county.index)
        total_mass = pd.Series(0.0, index=tr_for_county.index)
        for i in COMPOSITE_MEMBERS[key]:
            o = OFFENSES[i]
            c = pd.to_numeric(tr_for_county[f"expected_count_{o}"], errors="coerce").fillna(0.0)
            w = (
                tr_for_county[f"source_mode_{o}"].astype("string").map(SOURCE_MODE_CODE)
                .map(MODE_DIRECT_WEIGHT).fillna(0.0)
            )
            direct_mass += c * w
            total_mass += c
        g = tr_for_county["county_geoid"]
        ratio = (
            direct_mass.groupby(g).sum()
            / total_mass.groupby(g).sum().replace(0.0, np.nan)
        ).round(3)
        for cg, v in ratio.items():
            county_ds.setdefault(str(cg), {})[key] = None if pd.isna(v) else float(v)
    # Counties carry no per-offence detail of their own; the card for a county
    # shows only the rollup average, so the record keeps names and population.
    cvals = {m: county[m].to_numpy(dtype="float64") for m in MEASURE_ORDER}
    cids = county["county_geoid"].astype(str).to_numpy()
    cpop = county["pop"].to_numpy(dtype="float64")
    county_records = {
        cids[i]: [
            county_names.get(cids[i], ""),
            "",
            "",
            num(cpop[i], 0),
            None,
            0,
            [num(cvals[m][i], 1) for m in MEASURE_ORDER],
            None,
            None,
            None,
            None,
            None,
            [county_ds.get(cids[i], {}).get(k) for k in COMPOSITE_ORDER],
            # A county spans many agencies, so there is no one jurisdiction total
            # to describe; the county card says so in its own note instead.
            None,
        ]
        for i in range(len(county))
    }
    stats["county"] = shard(county_records, "county")
    print(f"  {stats['county']}")

    print("Tract shards")
    tr = pd.read_parquet(WORK / "tract_core.parquet")
    tract_level = level_total_reported(TRACT_SRC, "tract_id")
    stats["tract"] = shard(
        build_records(tr, "tract_id", tract_names, county_names, None, tract_level),
        "tract",
    )
    print(f"  {stats['tract']}")

    print("Block-group shards")
    bg = pd.read_parquet(WORK / "bg_core.parquet")
    bg_level = level_total_reported(BG_SRC, "block_group_geoid")
    stats["blockgroup"] = shard(
        build_records(
            bg, "block_group_geoid", bg_names, county_names, tract_names, bg_level
        ),
        "bg",
    )
    print(f"  {stats['blockgroup']}")

    print("Benchmark files")
    bench = json.loads((WORK / "benchmarks.json").read_text())
    measures = bench["measures"]
    by_state: dict[str, dict] = {}
    for jid, row in bench["jurisdiction"].items():
        ss = jid.split(":")[0]
        st = None
        from crschema import FIPS_TO_USPS

        st = FIPS_TO_USPS.get(ss)
        if st is None:
            continue
        entry = by_state.setdefault(st, {"state": bench["state"].get(st, {}), "j": {}})
        entry["j"][jid] = {
            "name": display_name(row["name"]),
            # 0 = below the support floor; the card drops the term rather than
            # printing a benchmark built from a single cell.
            "s": int(row.get("supported", 1)),
            "v": [row.get(m) for m in measures],
        }
    bench_bytes = 0
    for st, entry in by_state.items():
        entry["measures"] = measures
        entry["state_name"] = STATE_NAME.get(st, st)
        p = BENCH_DIR / f"{st}.json"
        p.write_text(json.dumps(entry, separators=(",", ":")))
        bench_bytes += p.stat().st_size
    stats["bench"] = {
        "states": len(by_state),
        "total_bytes": bench_bytes,
        "max_bytes": max(p.stat().st_size for p in BENCH_DIR.glob("*.json")),
    }
    print(f"  {stats['bench']}")

    (WORK / "shard_manifest.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
