"""05 - Build the public download package.

One tidy table per geography, nationwide and per state, in CSV (gzipped) and
Parquet, plus a per-state GeoParquet carrying the same rows with 2020 TIGER
geometry. Nothing outside the documented public schema is published, and no
incident-level record is published at any point.

  dist/site/downloads/crimerisk_2025_block_group.csv.gz | .parquet
  dist/site/downloads/crimerisk_2025_tract.csv.gz       | .parquet
  dist/site/downloads/state/crimerisk_2025_<geo>_<ST>.csv.gz | .parquet | .geoparquet
  dist/site/downloads/fields.csv          data dictionary
  dist/site/downloads/checksums.sha256    sha256 of every published file

Run:  uv run python frontend/build/05_download.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import (  # noqa: E402
    COMPOSITES,
    DATA_ROOT,
    DIST,
    OFFENSE_LABEL,
    OFFENSES,
    POPULATION_COL,
    PUBLIC_COMPOSITE_NAMES,
    PUBLIC_OFFENSE_FIELDS,
    WORK,
    YEAR,
)

SITE = DIST / "site"
OUT = SITE / "downloads"
STATE_OUT = OUT / "state"

GEOGRAPHIES = {
    "block_group": {
        "core": "bg_core.parquet",
        "id": "block_group_geoid",
        "ids": ["block_group_geoid", "tract_id", "state_fips", "state"],
        "tiger": (DATA_ROOT / "tiger_bg", "tl_2020_{ss}_bg.zip", 12),
    },
    "tract": {
        "core": "tract_core.parquet",
        "id": "tract_id",
        "ids": ["tract_id", "state_fips", "state"],
        "tiger": (DATA_ROOT / "tiger_tracts", "tl_2020_{ss}_tract.zip", 11),
    },
}

FIELD_DOC = {
    "block_group_geoid": "12-digit 2020 Census block group identifier.",
    "tract_id": "11-digit 2020 Census tract identifier.",
    "state_fips": "Two-digit state FIPS code.",
    "state": "Two-letter state postal abbreviation.",
    f"population_{YEAR}": f"Estimated resident population, {YEAR}.",
    "land_area_sq_mi": "Land area in square miles (2020 TIGER, water excluded).",
    "special_use_type": "Area character: ordinary, park_open_space, industrial_employment, institutional_facility, campus_institution, transient_destination, group_quarters_other, unknown_special_use.",
    "jurisdiction_id": "Identifier of the police jurisdiction whose reported totals anchor this area.",
    "jurisdiction_name": "Display name of that police jurisdiction.",
}

OFFENSE_DOC = {
    "expected_count": "Estimated {label} offences in {year}.",
    "primary_denominator": "Denominator used for the exposure rate (people, premises or vehicles present).",
    "rate_primary": "{label} offences per 100,000 units of the exposure denominator.",
    "index_primary": "{label} exposure index; 100 = U.S. average.",
    "rate_resident": "{label} offences per 100,000 residents.",
    "index_resident": "{label} per-resident index; 100 = U.S. average.",
    "source_mode": "direct_city_incident, mixed or modeled_transfer: where the {label} estimate came from.",
}

COMPOSITE_DOC = {
    "overall_crime_exposure": "All seven offences combined, exposure-adjusted; 100 = U.S. average.",
    "violent_crime_exposure": "Murder, rape, robbery and aggravated assault combined, exposure-adjusted; 100 = U.S. average.",
    "property_crime_exposure": "Burglary, larceny/theft and motor vehicle theft combined, exposure-adjusted; 100 = U.S. average.",
    "overall_per_resident": "All seven offences combined, per resident; 100 = U.S. average.",
    "violent_per_resident": "Violent offences combined, per resident; 100 = U.S. average.",
    "property_per_resident": "Property offences combined, per resident; 100 = U.S. average.",
}


def public_frame(core: pd.DataFrame, geo: str, juris_names: dict) -> pd.DataFrame:
    spec = GEOGRAPHIES[geo]
    df = pd.DataFrame(index=core.index)
    for c in spec["ids"]:
        if c == "state":
            df["state"] = core["st"]
        else:
            df[c] = core[c].astype(str)
    df[f"population_{YEAR}"] = core["pop"]
    df["land_area_sq_mi"] = core["land_area_sq_mi"].round(4)
    df["special_use_type"] = core["special_use_type"]
    df["jurisdiction_id"] = core["juris_id"]
    df["jurisdiction_name"] = core["juris_id"].map(juris_names)
    for o in OFFENSES:
        for src_tpl, out_tpl in PUBLIC_OFFENSE_FIELDS:
            src = src_tpl.format(o=o)
            out = out_tpl.format(o=o)
            v = core[src]
            if pd.api.types.is_numeric_dtype(v):
                v = v.round(4)
            df[out] = v
    for key, (expo, resi, _l) in COMPOSITES.items():
        df[PUBLIC_COMPOSITE_NAMES[(key, "exposure")]] = core[expo].round(1)
        df[PUBLIC_COMPOSITE_NAMES[(key, "resident")]] = core[resi].round(1)
    return df


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", newline="", compresslevel=6) as fh:
        df.to_csv(fh, index=False)


def fields_csv(df: pd.DataFrame, path: Path) -> None:
    rows = [("name", "meaning")]
    for c in df.columns:
        if c in FIELD_DOC:
            rows.append((c, FIELD_DOC[c]))
            continue
        if c in COMPOSITE_DOC:
            rows.append((c, COMPOSITE_DOC[c]))
            continue
        doc = None
        for o in OFFENSES:
            if c.startswith(o + "_"):
                suffix = c[len(o) + 1 :]
                if suffix in OFFENSE_DOC:
                    doc = OFFENSE_DOC[suffix].format(label=OFFENSE_LABEL[o], year=YEAR)
                break
        rows.append((c, doc or ""))
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def geoparquet_for_state(geo: str, ss: str, part: pd.DataFrame, id_col: str, path: Path) -> None:
    tiger_dir, tpl, id_len = GEOGRAPHIES[geo]["tiger"]
    g = gpd.read_file(f"zip://{tiger_dir / tpl.format(ss=ss)}")
    g["GEOID"] = g["GEOID"].astype(str).str.zfill(id_len)
    g = g[["GEOID", "geometry"]].to_crs(4326)
    merged = g.merge(part, left_on="GEOID", right_on=id_col, how="inner").drop(columns=["GEOID"])
    gpd.GeoDataFrame(merged, geometry="geometry", crs=4326).to_parquet(
        path, compression="zstd", index=False
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_OUT.mkdir(parents=True, exist_ok=True)
    bench = json.loads((WORK / "benchmarks.json").read_text())
    juris_names = {k: v["name"] for k, v in bench["jurisdiction"].items()}

    summary = {}
    for geo, spec in GEOGRAPHIES.items():
        print(f"{geo}: building public table")
        core = pd.read_parquet(WORK / spec["core"])
        core["state_fips"] = core["state_fips"].astype(str).str.zfill(2)
        df = public_frame(core, geo, juris_names)
        print(f"  {len(df):,} rows x {len(df.columns)} columns")

        base = OUT / f"crimerisk_{YEAR}_{geo}"
        df.to_parquet(base.with_suffix(".parquet"), compression="zstd", index=False)
        write_csv_gz(df, Path(str(base) + ".csv.gz"))
        fields_csv(df, OUT / f"fields_{geo}.csv")

        # Nationwide GeoParquet, written state by state so peak memory stays low.
        geo_path = OUT / f"crimerisk_{YEAR}_{geo}.geoparquet"
        writer = None
        for ss, part in df.groupby("state_fips", sort=True):
            st = part["state"].iloc[0]
            sbase = STATE_OUT / f"crimerisk_{YEAR}_{geo}_{st}"
            part.to_parquet(sbase.with_suffix(".parquet"), compression="zstd", index=False)
            write_csv_gz(part, Path(str(sbase) + ".csv.gz"))
            gp = Path(str(sbase) + ".geoparquet")
            geoparquet_for_state(geo, ss, part, spec["id"], gp)
            tbl = pq.read_table(gp)
            if writer is None:
                writer = pq.ParquetWriter(
                    geo_path, tbl.schema, compression="zstd",
                )
            writer.write_table(tbl.cast(writer.schema))
            print(f"    {st}", end="", flush=True)
        if writer is not None:
            writer.close()
        print()
        summary[geo] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "parquet_bytes": base.with_suffix(".parquet").stat().st_size,
            "csv_gz_bytes": Path(str(base) + ".csv.gz").stat().st_size,
            "geoparquet_bytes": geo_path.stat().st_size,
        }

    print("Checksums")
    lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256":
            lines.append(f"{sha256(p)}  {p.relative_to(OUT)}")
    (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n")
    summary["files"] = len(lines)
    (WORK / "download_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
