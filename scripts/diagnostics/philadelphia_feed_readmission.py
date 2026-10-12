"""Evidence for re-adjudicating Philadelphia's blocked feed offenses (2025.3).

    uv run python scripts/diagnostics/philadelphia_feed_readmission.py [--out state/review/packets/city/philadelphia/share_admission]

Per offense and year, from the release feed's own loader path (`_prepare_philadelphia_incidents`, the
packet crosswalk with every blocked offense switched on) and the same block-group join the share
surface uses:
- feed counts: mapped, geocoded, matched to a Philadelphia block group;
- the FBI-published count for PAPEP0000 (agency_year_observations, highest-weight row, months reported);
- the PPD "Major Crimes as reported to PPD" preliminary year-to-date table
  (data/PPD-Crime-Statistics, scripts/pull/pull_ppd_crime_statistics.py);
- per police district (from the DC number, digits 5-6): feed counts, matched share, and the PPD district
  table, with the total variation distance between the two district distributions.
Writes reconciliation.csv, districts.csv, district_tvd.csv and qa.csv (the share-admission QA table:
mapped rows, rows matched to a Philadelphia block group, the FBI count) to --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from crimerisk import city_incidents as ci  # noqa: E402

OFFENSES = ["murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny", "motor_vehicle_theft"]
PPD_TYPES = {
    "murder": ["Homicide"],
    "rape": ["Rape"],
    "robbery": ["Robbery/Gun", "Robbery/Other"],
    "aggravated_assault": ["Aggravated Assault/Gun", "Aggravated Assault/Other"],
    "burglary": ["Burglary/Residential", "Burglary/Commercial"],
    "larceny": ["Theft from Person", "Theft from Auto", "Theft", "Theft/Retail"],
    "motor_vehicle_theft": ["Auto Theft"],
}
ORI = "PAPEP0000"
# The PPD district tables report the 6th district as empty from 2023 (its id returns no rows) and
# the 9th's totals match the feed's 6th + 9th; the two are compared as one district.
DISTRICT_MERGE = {"06": "09"}


def feed(start: int, end: int) -> pd.DataFrame:
    packet = REPO / "state" / "review" / "packets" / "city" / "philadelphia"
    crosswalk = pd.read_csv(packet / "offense_crosswalk.csv", dtype=str)
    # Every row the packet maps to one of the seven offenses, blocked ones included (justifiable
    # homicide stays out: it is not murder).
    crosswalk.loc[crosswalk["v2_offense"].isin(OFFENSES) & ~crosswalk["source_value"].str.contains("Justifiable"), "include_in_v2"] = "yes"
    tmp = REPO / "state" / "qa" / "_philadelphia_crosswalk_all.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(tmp, index=False)
    raw = ci._load_philadelphia_raw_snapshot(raw_dir=REPO / "data" / "city-incidents" / "philadelphia", start_year=start, end_year=end)
    raw["year"] = pd.to_datetime(raw["dispatch_date_time"], errors="coerce").dt.year
    raw["ucr_general"] = raw["ucr_general"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    raw["text_general_code"] = raw["text_general_code"].astype(str).str.strip()
    raw["offense"] = pd.NA
    for row in ci._load_philadelphia_offense_crosswalk(tmp).itertuples(index=False):
        mask = pd.Series(True, index=raw.index)
        for field, value in zip(row.source_fields, row.source_values, strict=True):
            mask &= raw[field].astype(str).eq(value)
        raw.loc[mask, "offense"] = row.offense
    tmp.unlink()
    mapped = raw[raw["offense"].notna() & raw["year"].between(start, end)].copy()
    mapped["district"] = mapped["dc_key"].astype(str).str[4:6]
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    xw = pd.read_parquet(REPO / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet", columns=["block_group_geoid", "jurisdiction_id"])
    phl = set(xw.loc[xw["jurisdiction_id"].eq(ci.PHILADELPHIA_JURISDICTION_ID), "block_group_geoid"].astype(str).str.zfill(12))
    bg = gpd.read_file(REPO / "data" / "tiger_bg" / "tl_2020_42_bg.zip")[["GEOID", "geometry"]]
    bg = bg[bg["GEOID"].isin(phl)].to_crs("EPSG:4326")
    located = mapped[mapped["latitude"].notna() & mapped["longitude"].notna()]
    pts = gpd.GeoDataFrame(located[[]], geometry=gpd.points_from_xy(located["longitude"], located["latitude"]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, bg, how="inner", predicate="within")
    mapped["geocoded"] = mapped["latitude"].notna() & mapped["longitude"].notna()
    mapped["matched"] = mapped.index.isin(joined.index)
    return mapped


def official() -> pd.DataFrame:
    obs = pd.read_parquet(REPO / "state" / "observations" / "agency_year_observations.parquet",
                          columns=["ori9", "year", "offense", "source", "count", "observation_weight", "months_reported"])
    obs = obs[obs["ori9"].eq(ORI)].copy()
    obs["offense"] = obs["offense"].astype(str)
    obs = obs.sort_values(["observation_weight", "count"], ascending=False).drop_duplicates(["year", "offense"])
    return obs.rename(columns={"count": "fbi_count", "source": "fbi_source", "months_reported": "fbi_months"})[
        ["year", "offense", "fbi_count", "fbi_source", "fbi_months"]]


def ppd(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"district": "string"})
    # Latest table per (scope, district, year): a later as-of date carries the revised prior year.
    table = table.sort_values("as_of_date").drop_duplicates(["scope", "district", "year", "crime_type"], keep="last")
    rows = []
    for offense, types in PPD_TYPES.items():
        sub = table[table["crime_type"].isin(types)]
        g = sub.groupby(["scope", "district", "year"], dropna=False)["count"].sum().reset_index()
        g["offense"] = offense
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["district"] = out["district"].fillna("").map(lambda d: f"{int(float(d)):02d}" if d else "")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--ppd", default=str(REPO / "data" / "PPD-Crime-Statistics" / "ppd_major_crimes.csv"))
    parser.add_argument("--out", default=str(REPO / "state" / "review" / "packets" / "city" / "philadelphia" / "share_admission"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = feed(args.start, args.end)
    p = ppd(Path(args.ppd))

    recon = f.groupby(["year", "offense"]).agg(mapped=("objectid", "size"), geocoded=("geocoded", "sum"), matched=("matched", "sum")).reset_index()
    recon["matched_share"] = recon["matched"] / recon["mapped"]
    recon = recon.merge(official(), on=["year", "offense"], how="left")
    city = p[p["scope"].eq("citywide")][["year", "offense", "count"]].rename(columns={"count": "ppd_table"})
    recon = recon.merge(city, on=["year", "offense"], how="left")
    recon["feed_vs_fbi"] = recon["mapped"] / recon["fbi_count"] - 1
    recon["feed_vs_ppd_table"] = recon["mapped"] / recon["ppd_table"] - 1
    recon["fbi_vs_ppd_table"] = recon["fbi_count"] / recon["ppd_table"] - 1
    recon.to_csv(out / "reconciliation.csv", index=False)
    recon.assign(official_source=recon["fbi_source"] + ":" + recon["fbi_months"].astype("Int64").astype(str) + "_months").rename(
        columns={"mapped": "mapped_rows", "matched": "located_rows", "fbi_count": "official_total"})[
        ["year", "offense", "mapped_rows", "located_rows", "official_total", "official_source"]].to_csv(out / "qa.csv", index=False)

    f["district"] = f["district"].replace(DISTRICT_MERGE)
    d = f.groupby(["year", "offense", "district"]).agg(feed=("objectid", "size"), matched=("matched", "sum")).reset_index()
    d["matched_share"] = d["matched"] / d["feed"]
    dist = p[p["scope"].eq("district")].assign(district=lambda t: t["district"].replace(DISTRICT_MERGE))
    dist = dist.groupby(["year", "offense", "district"], as_index=False)["count"].sum().rename(columns={"count": "ppd_table"})
    d = d.merge(dist, on=["year", "offense", "district"], how="outer")
    d.to_csv(out / "districts.csv", index=False)

    rows = []
    for (year, offense), g in d.groupby(["year", "offense"]):
        both = g[g["ppd_table"].notna() & g["feed"].notna()]
        low = g[(g["feed"].fillna(0) >= 20) & (g["matched_share"] < 0.80)]
        a = both["feed"] / max(both["feed"].sum(), 1)
        b = both["ppd_table"] / max(both["ppd_table"].sum(), 1)
        ratio = both["feed"] / both["ppd_table"].replace(0, np.nan)
        rows.append({"year": int(year), "offense": offense, "districts_compared": int(len(both)),
                     "feed_rows_in_compared": int(both["feed"].sum()), "feed_rows_other_districts": int(g.loc[g["ppd_table"].isna(), "feed"].sum()),
                     "ppd_district_sum": int(both["ppd_table"].sum()),
                     "tvd_district_shares": float((a - b).abs().sum() / 2) if len(both) else np.nan,
                     "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
                     "min_district_matched_share": float(g.loc[g["feed"].fillna(0) >= 20, "matched_share"].min()),
                     "districts_below_080": ";".join(f"{r.district}:{r.matched_share:.2f}/{int(r.feed)}" for r in low.itertuples())})
    tvd = pd.DataFrame(rows)
    tvd.to_csv(out / "district_tvd.csv", index=False)
    pd.set_option("display.width", 220)
    print(recon.round(3).to_string(index=False))
    print(tvd.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
