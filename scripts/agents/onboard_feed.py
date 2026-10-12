"""City feed onboarding (design section 7.2).

    uv run python scripts/agents/onboard_feed.py --city buffalo --city tacoma --from-e4 --year-end 2025 --run <run_id> [--mode session]
    uv run python scripts/agents/onboard_feed.py --city chandler_az --new --year-end 2025 --run <run_id> [--mode session]
    uv run python scripts/agents/onboard_feed.py --seal 20260927 --run <run_id>
    uv run python scripts/agents/onboard_feed.py ... --dry-run

Per city: D init packet -> O contract draft -> R review (one revise round) -> D replay of the
contract on the archived pull against the E4 `incidents_bg.parquet` (1% per year and offense, 0.5%
block-group assignment; one fix round with the diff) -> D live extension to --year-end -> D QA ->
O/R `feed_exception` admission when QA has exceptions, else the truth_only rule -> proposals for
`ledger.py approve`. State lives under `state/review/runs/<run_id>/<city>/`; rerunning continues.
In session mode a pending agent step writes `<step>.task.md` and the city waits (exit code 2).

`--new` qualifies a source that has no E4 oracle, from its row in `configs/city_source_registry.csv`:
D init packet (portal metadata and docs archived into `evidence/`, the 2018..year-end pull archived as
`raw_cache.parquet`, official totals for the publishing ORIs or the scope jurisdictions) -> O contract
draft -> D qualification (the contract runs on the archived pull; the loss funnel raw -> mapped ->
dated -> in_years -> located -> in_state -> in_footprint -> deduplicated is counted before any
filter; one fix round when the contract errors, maps nothing, or places under half of its in-year
rows in a block group) -> R review (a `revise` the reviewer would not sign gets one drafter round and a fresh review) -> D QA, where reconciliation to the
official totals is a material check -> O/R batched admission -> proposals. Truth rows are the contract's own located incidents.

`--seal <seed>` draws one third of the run's new sources with approved truth_only admissions for
final evaluation, stratified by Census region and population class, and writes one
`evaluation_seal` proposal per sealed source (`configs/truth_source_seals.csv`). Sealed sources go to
`truth_surface_sealed_2018_2025.parquet`, never to the training truth surface.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import REPO_ROOT, RUNS_DIR, SKILLS_DIR  # noqa: E402

from crimerisk import city_contracts as cc  # noqa: E402
from crimerisk import level_lane  # noqa: E402
from crimerisk.paths import get_paths  # noqa: E402
from crimerisk.state_publications import drop_definition_mismatches  # noqa: E402

SKILL = SKILLS_DIR / "onboard-source"
E4_ROOT = REPO_ROOT / "analysis_scratch" / "final_phase" / "corpus_expansion"
PACKETS = REPO_ROOT / "state" / "review" / "packets" / "city"
CROSSWALK = REPO_ROOT / "state" / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet"
OBSERVATIONS = REPO_ROOT / "state" / "observations" / "jurisdiction_year_observations.parquet"
TIGER = REPO_ROOT / "data" / "tiger_bg"
E4_YEARS = (2018, 2024)
TRUTH_START = 2018
REGISTRY = REPO_ROOT / "configs" / "city_source_registry.csv"
SEALS = REPO_ROOT / "configs" / "truth_source_seals.csv"
AGENCY_OBSERVATIONS = REPO_ROOT / "state" / "observations" / "agency_year_observations.parquet"
JURISDICTIONS = REPO_ROOT / "state" / "reference" / "jurisdiction_master.parquet"
CENSUS_REGION = {  # state FIPS -> Census region
    **{f: "northeast" for f in ("09", "23", "25", "33", "34", "36", "42", "44", "50")},
    **{f: "midwest" for f in ("17", "18", "19", "20", "26", "27", "29", "31", "38", "39", "46", "55")},
    **{f: "south" for f in ("01", "05", "10", "11", "12", "13", "21", "22", "24", "28", "37", "40", "45", "47", "48", "51", "54")},
    **{f: "west" for f in ("02", "04", "06", "08", "15", "16", "30", "32", "35", "41", "49", "53", "56")},
}
SIZE_CLASSES = ((250_000, "250k+"), (100_000, "100k-250k"), (0, "under 100k"))
CAD_SOURCES = {"nola", "ottawa_mi"}
APPROVER = "claude-pm-session (delegated by owner)"
# QA checks that can bar truth_only and go to the drafter/reviewer; the others (reconciliation to
# FBI totals, a failed live extension) are recorded and do not bar truth_only for 2018-2024.
MATERIAL_CHECKS = {"match_share", "exact_point", "year_coverage", "contract_replay"}
# A new source has no oracle, so the official totals are the only outside check on its mapping and
# coverage: reconciliation is material there, and so is a contract that does not qualify.
MATERIAL_CHECKS_NEW = MATERIAL_CHECKS | {"reconciliation", "contract_qualification"}
# A new source whose qualified contract keeps fewer rows than this in its footprint over the whole
# window is rejected by rule before review.
MIN_FOOTPRINT_ROWS = 1000


class NeedsBatch(Exception):
    """The city has material QA exceptions and waits for the run's batched admission case."""


def render(name: str, **values: str) -> str:
    text = (SKILL / "prompts" / name).read_text()
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


# ------------------------------------------------------------------ D: init


def _profile(raw: pd.DataFrame) -> dict:
    columns = []
    for name in raw.columns:
        series = raw[name]
        entry = {"name": str(name), "dtype": str(series.dtype), "non_null": int(series.notna().sum())}
        if series.dtype == object or str(series.dtype).startswith("string"):
            counts = series.astype("string").value_counts(dropna=True)
            if len(counts) <= 300:
                entry["distinct"] = {str(k): int(v) for k, v in counts.items()}
            else:
                entry["distinct_count"] = int(len(counts))
                entry["examples"] = [str(v) for v in counts.index[:5]]
        else:
            entry["min"] = None if series.dropna().empty else str(series.min())
            entry["max"] = None if series.dropna().empty else str(series.max())
        columns.append(entry)
    return {"rows": int(len(raw)), "columns": columns}


def init_packet(city: str) -> tuple[Path, str]:
    src = E4_ROOT / city
    packet = PACKETS / city
    evidence = packet / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src.iterdir()) if p.name.startswith("01_fetch") or p.name in {"README.md", "offense_mapping.csv"}]
    for p in files:
        shutil.copy2(p, evidence / p.name)
    raw_target = packet / "raw_cache.parquet"
    if not raw_target.exists():
        shutil.copy2(src / "raw_cache.parquet", raw_target)
    raw = pd.read_parquet(raw_target)
    (evidence / "raw_cache_profile.json").write_text(json.dumps(_profile(raw), indent=1) + "\n")
    widened = pd.read_parquet(E4_ROOT / "widened_incident_surface.parquet")
    jur = (widened[widened["source"].eq(city)].groupby("jurisdiction_id", dropna=False)["incident_count"].sum()
           .sort_values(ascending=False).rename("incidents_2018_2024").reset_index())
    names = pd.read_parquet(REPO_ROOT / "state" / "reference" / "jurisdiction_master.parquet", columns=["jurisdiction_id", "jurisdiction_name"])
    jur = jur.merge(names.drop_duplicates("jurisdiction_id"), on="jurisdiction_id", how="left")
    jur.to_csv(evidence / "jurisdictions.csv", index=False)
    evidence_files = sorted(p for p in evidence.iterdir() if p.is_file()) + [raw_target]
    case_text = (
        f"Source {city}: an E4 truth-corpus city incident feed (2018-2024), fetched by "
        f"evidence/{next((p.name for p in files if p.name.startswith('01_fetch')), 'n/a')}. "
        f"README present: {'yes' if (src / 'README.md').exists() else 'no'}; offense_mapping.csv present: "
        f"{'yes' if (src / 'offense_mapping.csv').exists() else 'no'}."
    )
    sha = common.write_packet(packet, case_text=case_text, signals={}, evidence=evidence_files)
    return packet, sha


# ------------------------------------------------------------------ D: replay, extension, QA


def oracle(city: str) -> pd.DataFrame:
    frame = pd.read_parquet(E4_ROOT / city / "incidents_bg.parquet")
    keep = [c for c in ("incident_id", "year", "offense", "block_group_geoid", "latitude", "longitude") if c in frame.columns]
    frame = frame[keep].copy()
    frame = frame[frame["year"].between(*E4_YEARS)]
    frame["block_group_geoid"] = frame["block_group_geoid"].astype("string")
    return frame


def replay(contract: dict, packet: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_parquet(packet / "raw_cache.parquet")
    try:
        incidents = cc.run_contract(contract, raw, year_start=E4_YEARS[0], year_end=E4_YEARS[1])
    except Exception as exc:  # noqa: BLE001 - a contract that cannot run is a failed replay
        incidents = pd.DataFrame(columns=["incident_id", "year", "offense", "latitude", "longitude"])
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    else:
        error = ""
    if not incidents.empty:
        incidents = cc.assign_block_groups(incidents, contract["geography"]["state_fips"], TIGER)
    else:
        incidents["block_group_geoid"] = pd.Series(dtype="string")
    counts, summary = cc.replay_check(incidents, oracle(contract["city_key"]))
    if error:
        summary.update(pass_=False, error=error)
        summary["pass"] = False
        summary.pop("pass_")
    return counts, summary, incidents


def extend(contract: dict, year: int, packet: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        raw = cc.download(contract, year, year, packet / "live")
    except Exception as exc:  # noqa: BLE001 - the failure is recorded as a QA exception
        return None, f"download failed: {type(exc).__name__}: {str(exc)[:300]}"
    try:
        incidents = cc.run_contract(contract, raw, year_start=year, year_end=year)
    except Exception as exc:  # noqa: BLE001
        columns = sorted(map(str, raw.columns))
        return None, f"contract does not map the live pull: {type(exc).__name__}: {str(exc)[:200]}; live columns {columns}"
    if incidents.empty:
        return None, f"live pull returned {len(raw)} rows and no mapped rows for {year}"
    return cc.assign_block_groups(incidents, contract["geography"]["state_fips"], TIGER), ""


def _official(jurisdictions: list[str]) -> pd.DataFrame:
    """The scope jurisdictions' official counts, definition mismatches dropped. These are agency
    aggregates, so the per-ORI level-lane rulings (`admitted_agency_counts`) cannot be applied here;
    a registry row naming the publishing ORIs gets them."""
    obs = pd.read_parquet(OBSERVATIONS, columns=["jurisdiction_id", "year", "offense", "source", "raw_data_source",
                                                 "observed_count", "observation_weight"])
    obs = drop_definition_mismatches(obs[obs["jurisdiction_id"].isin(jurisdictions)]).copy()
    obs["offense"] = obs["offense"].astype(str)
    obs = obs.sort_values(["observation_weight", "observed_count"], ascending=False)
    return obs.drop_duplicates(["jurisdiction_id", "year", "offense"]).rename(
        columns={"observed_count": "official_total", "source": "official_source"})


def qa(city: str, contract: dict | None, incidents: pd.DataFrame, primary: str, *, funnel_table: pd.DataFrame | None = None,
       official: pd.DataFrame | None = None, first_year: int = E4_YEARS[0]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per offense and year QA rows, and the rows that fail a threshold.

    With `funnel_table` (new sources), `mapped_rows` and `located_rows` are the funnel's in_years and
    in_state counts, taken before the coordinate filter, so `match_share` is the share of in-year
    mapped rows that land in a block group; `footprint_rows` (the funnel's deduplicated count) is
    reconciled against `official` (year, offense, official_total, official_source).
    """
    rows = []
    xw = pd.read_parquet(CROSSWALK, columns=["block_group_geoid", "jurisdiction_id"])
    primary_bgs = set(xw.loc[xw["jurisdiction_id"].eq(primary), "block_group_geoid"].astype(str).str.zfill(12))
    for (year, offense), g in incidents.groupby(["year", "offense"]):
        located = g[g["block_group_geoid"].notna()]
        exact = 0.0
        if len(located) and {"latitude", "longitude"} <= set(located.columns):
            pts = located[["latitude", "longitude"]].round(6).value_counts()
            exact = float(pts.iloc[0] / len(located)) if len(pts) else 0.0
        in_primary = located["block_group_geoid"].astype(str).str.zfill(12).isin(primary_bgs)
        rows.append({"city_key": city, "year": int(year), "offense": offense, "mapped_rows": int(len(g)),
                     "located_rows": int(len(located)), "match_share": len(located) / len(g) if len(g) else np.nan,
                     "primary_jurisdiction_rows": int(in_primary.sum()), "exact_point_max_share": exact})
    table = pd.DataFrame(rows, columns=["city_key", "year", "offense", "mapped_rows", "located_rows", "match_share",
                                        "primary_jurisdiction_rows", "exact_point_max_share"])
    feed_column = "primary_jurisdiction_rows"
    if funnel_table is not None:
        yearly = funnel_table[funnel_table["year"].ne("all") & funnel_table["offense"].isin(cc.OFFENSES_7)].copy()
        yearly["year"] = yearly["year"].astype(int)
        yearly = yearly.rename(columns={"in_years": "mapped_rows", "in_state": "located_rows", "deduplicated": "footprint_rows"})
        yearly = yearly[["year", "offense", "mapped_rows", "located_rows", "footprint_rows"]]
        table = yearly.merge(table.drop(columns=["mapped_rows", "located_rows", "match_share"]), on=["year", "offense"], how="left")
        table["city_key"] = city
        table["primary_jurisdiction_rows"] = table["primary_jurisdiction_rows"].fillna(0).astype(int)
        table["exact_point_max_share"] = table["exact_point_max_share"].fillna(0.0)
        table["match_share"] = table["located_rows"] / table["mapped_rows"].replace(0, np.nan)
        table = table[["city_key", "year", "offense", "mapped_rows", "located_rows", "match_share", "primary_jurisdiction_rows",
                       "footprint_rows", "exact_point_max_share"]].sort_values(["year", "offense"]).reset_index(drop=True)
        feed_column = "footprint_rows"
    if official is None:
        official = _official([primary])
    table = table.merge(official[["year", "offense", "official_total", "official_source"]], on=["year", "offense"], how="left")
    table["reconciliation_relative_delta"] = (
        (table[feed_column] - table["official_total"]).abs() / table["official_total"].replace(0, np.nan))
    years = set(range(first_year, int(table["year"].max()) + 1)) if not table.empty else set()
    checks = []
    for _, r in table.iterrows():
        if r["match_share"] < cc.MIN_MATCH_SHARE:
            checks.append((r["offense"], r["year"], "match_share", f"{r['match_share']:.3f} < {cc.MIN_MATCH_SHARE}"))
        if pd.notna(r["reconciliation_relative_delta"]) and r["reconciliation_relative_delta"] > cc.RECONCILIATION_TOLERANCE:
            checks.append((r["offense"], r["year"], "reconciliation",
                           f"feed {int(r[feed_column])} vs FBI {r['official_total']:.0f} ({r['reconciliation_relative_delta']:.2f})"))
    # Exact-point tripwire as in scripts/diagnostics/build_city_feed_exact_point_qa.py: per offense,
    # the largest share of located incidents at one coordinate over all years carried.
    located_all = incidents[incidents["block_group_geoid"].notna()]
    if {"latitude", "longitude"} <= set(located_all.columns):
        for offense, g in located_all.groupby("offense"):
            pts = g[["latitude", "longitude"]].round(6).value_counts()
            share = float(pts.iloc[0] / len(g)) if len(pts) else 0.0
            if share >= cc.EXACT_POINT_SHARE and len(g) >= 200:
                lat, lon = pts.index[0]
                checks.append((offense, "", "exact_point", f"{share:.4f} of {len(g)} located incidents at ({lat}, {lon})"))
    for offense in cc.OFFENSES_7:
        present = set(table.loc[table["offense"].eq(offense) & table["located_rows"].gt(0), "year"]) if not table.empty else set()
        missing = sorted(years - present)
        if missing:
            checks.append((offense, "", "year_coverage", f"no rows in {missing}"))
    exceptions = pd.DataFrame(checks, columns=["offense", "year", "check", "detail"])
    exceptions.insert(0, "city_key", city)
    return table, exceptions


# ------------------------------------------------------------------ new sources: registry, init, qualification


def registry_row(city: str) -> dict:
    registry = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
    rows = registry[registry["city_key"].eq(city)]
    if len(rows) != 1:
        raise SystemExit(f"{common.rel(REGISTRY)}: expected one row for {city}, found {len(rows)}")
    return rows.iloc[0].to_dict()


def _split(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(";") if v.strip()]


def probe_sources(row: dict) -> list[dict]:
    """The registry's endpoints as contract `sources`: select everything, filter only by date."""
    endpoints = _split(row["api_endpoint"])
    wheres = _split(row["probe_where"]) or [""]
    return [{"slug": f"{row['city_key']}_{i}" if len(endpoints) > 1 else row["city_key"], "name": row["source_name"],
             "source_url": row["source_url"], "api_endpoint": endpoint, "portal_type": row["portal_type"], "select": [],
             "where": wheres[i] if len(wheres) == len(endpoints) else wheres[0], "order": row.get("probe_order", ""),
             "page_size": int(row.get("page_size") or (1000 if row["portal_type"].startswith("arcgis") else 50000))}
            for i, endpoint in enumerate(endpoints)]


def _fetch_text(url: str) -> bytes:
    from urllib.request import Request, urlopen

    with urlopen(Request(url, headers={"User-Agent": "crimerisk-city-contracts"}), timeout=120) as response:
        return response.read()


def _metadata_urls(source: dict) -> list[str]:
    endpoint = source["api_endpoint"]
    if source["portal_type"] == "socrata" and "/resource/" in endpoint:
        domain, dataset = endpoint.split("/resource/")
        return [f"{domain}/api/views/{dataset.split('.')[0]}.json"]
    if source["portal_type"].startswith("arcgis"):
        return [endpoint.rstrip("/").removesuffix("/query") + "?f=json"]
    return []


# Level-lane statuses whose count the build takes as an observed complete year. Any other ruling
# (coverage defective, identity failure, a partial lower bound, an omitted channel, a wrong or
# swapped category, unresolved review) leaves no comparator unless it carries a replacement count.
_COMPARATOR_LEVEL1 = frozenset({"valid_complete_year", "corroborated_structural_zero"})


def admitted_agency_counts(obs: pd.DataFrame, paths=None) -> pd.DataFrame:
    """The agency counts the build admits, as candidate comparators (ori9, year, offense, source,
    count, observation_weight, raw_data_source).

    A count is a candidate only if the build would read it: (1) a definition mismatch
    (`state_publications.drop_definition_mismatches`: FDLE and MS TOPS rape) is dropped; (2) an
    agency, year and offense with a level-lane hard-evidence row or a reporting-regime restriction
    in force (`level_lane.load_hard_evidence`) keeps no filed count, and its replacement count, when
    the row carries one, is the only candidate (source `level_lane_replacement:<case_id>`) unless
    the ruling accepts the filed count as a complete year; (3) an admission-registry `refuse_repair`
    or `refuse_silent` ruling drops the agency-year-offense; a wildcard ruling, which the build
    applies to the year's signaled offenses, drops the whole agency-year here, the signal not being reproduced.
    Screens the build computes from the whole panel (near-zero vectors, coverage by another ORI,
    the soft plausibility policy) are not reproduced here.
    """
    paths = paths or get_paths()
    keys = ["ori9", "year", "offense"]
    obs = drop_definition_mismatches(obs).copy()
    obs["ori9"] = obs["ori9"].astype(str).str.upper()
    obs["offense"] = obs["offense"].astype(str)
    obs["year"] = obs["year"].astype(int)
    rulings = []
    for year in sorted(obs["year"].unique()):
        hard = level_lane._expand_contract(level_lane.load_hard_evidence(paths, year=int(year)))
        if not hard.empty:
            rulings.append(hard.assign(year=int(year)))
    registry = level_lane.load_admission_registry(paths)
    registry = registry[registry["adjudication"].isin({"refuse_repair", "refuse_silent"})]
    refused = level_lane._expand_contract(registry)[keys] if not registry.empty else pd.DataFrame()
    if rulings:
        hard = pd.concat(rulings, ignore_index=True)
        hard = hard[hard["ori9"].isin(set(obs["ori9"]))]
        replacement = pd.to_numeric(hard["replacement_count"], errors="coerce")
        accepted = (hard["level1_status"].isin(_COMPARATOR_LEVEL1) & ~hard["level2_status"].isin(level_lane.INVALID_LEVEL2)
                    & replacement.isna())
        ruled = hard.loc[~accepted, keys]
        obs = obs.merge(ruled.assign(_ruled=True), on=keys, how="left")
        obs = obs[obs["_ruled"].isna()].drop(columns="_ruled")
        replaced = hard[replacement.notna()]
        if not replaced.empty:
            obs = pd.concat([obs, pd.DataFrame({
                "ori9": replaced["ori9"], "year": replaced["year"].astype(int), "offense": replaced["offense"].astype(str),
                "source": "level_lane_replacement:" + replaced["case_id"].astype(str), "raw_data_source": "level_lane_hard_evidence",
                "count": pd.to_numeric(replaced["replacement_count"], errors="coerce"), "observation_weight": np.inf})],
                ignore_index=True)
    if not refused.empty:
        refused = refused.drop_duplicates().assign(year=lambda f: f["year"].astype(int))
        obs = obs.merge(refused.assign(_refused=True), on=keys, how="left")
        obs = obs[obs["_refused"].isna()].drop(columns="_refused")
    return obs.reset_index(drop=True)


def official_totals(row: dict, jurisdiction_ids: list[str], paths=None) -> pd.DataFrame:
    """Official annual totals the feed is reconciled to: the publishing ORIs' own reports when the
    registry names them (one agency's records), else the scope jurisdictions' observations. Per
    ORI, year and offense the comparator is the admitted count (`admitted_agency_counts`) with the
    highest observation weight, then the highest count."""
    oris = _split(row.get("ori9", ""))
    if oris:
        obs = pd.read_parquet(AGENCY_OBSERVATIONS, columns=["ori9", "year", "offense", "source", "raw_data_source", "count",
                                                            "observation_weight"])
        obs = admitted_agency_counts(obs[obs["ori9"].isin(oris)], paths)
        obs = obs.sort_values(["observation_weight", "count"], ascending=False).drop_duplicates(["ori9", "year", "offense"])
        out = obs.groupby(["year", "offense"], as_index=False).agg(
            official_total=("count", "sum"), official_source=("source", lambda v: ";".join(sorted(set(map(str, v))))))
        out["official_basis"] = "ori:" + ";".join(oris)
        return out
    obs = _official(jurisdiction_ids)
    out = obs.groupby(["year", "offense"], as_index=False).agg(
        official_total=("official_total", "sum"), official_source=("official_source", lambda v: ";".join(sorted(set(map(str, v))))))
    out["official_basis"] = "jurisdictions:" + ";".join(jurisdiction_ids)
    return out


def init_new_packet(city: str, year_end: int) -> tuple[Path, str, dict]:
    row = registry_row(city)
    packet = PACKETS / city
    evidence = packet / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "registry_row.json").write_text(json.dumps(row, indent=1, sort_keys=True) + "\n")
    sources = probe_sources(row)
    urls = [u for src in sources for u in _metadata_urls(src)] + [row["source_url"]] + _split(row.get("docs_urls", ""))
    names = {url: f"doc_{common.sha256_text(url)[:10]}" for url in urls}
    for stale in evidence.glob("doc_*"):
        if stale.name != "doc_index.csv" and stale.name.split(".")[0] not in names.values():
            stale.unlink()
    for url, name in names.items():
        target = evidence / f"{name}.{'json' if 'f=json' in url or url.endswith('.json') else 'html'}"
        if not target.exists() and not target.with_suffix(".error.txt").exists():
            try:
                target.write_bytes(_fetch_text(url))
            except Exception as exc:  # noqa: BLE001 - an unreachable page is recorded, not fatal
                target.with_suffix(".error.txt").write_text(f"{url}\n{type(exc).__name__}: {exc}\n")
    (evidence / "doc_index.csv").write_text("file,url\n" + "".join(f"{name},{url}\n" for url, name in names.items()))
    raw_target = packet / "raw_cache.parquet"
    if not raw_target.exists():
        raw = cc.download({"sources": sources}, TRUTH_START, year_end, packet / "live")
        # Point objects (Socrata location, GeoJSON) are kept as JSON text; `field#point` reads either.
        for column in raw.columns[raw.dtypes.eq(object)]:
            nested = raw[column].map(lambda v: isinstance(v, (dict, list, np.ndarray)))
            if nested.any():
                raw[column] = raw[column].map(lambda v: json.dumps(v, default=lambda o: o.tolist())
                                              if isinstance(v, (dict, list)) else json.dumps(v.tolist()) if isinstance(v, np.ndarray) else v)
        try:
            raw.to_parquet(raw_target, index=False)
        except Exception:  # noqa: BLE001 - mixed-type portal columns are stored as text, as in cc.download
            mixed = [c for c in raw.columns if raw[c].dtype == object]
            raw[mixed] = raw[mixed].astype("string")
            raw.to_parquet(raw_target, index=False)
    raw = pd.read_parquet(raw_target)
    (evidence / "raw_cache_profile.json").write_text(json.dumps(_profile(raw), indent=1) + "\n")
    scope = _split(row["jurisdiction_ids"])
    names = pd.read_parquet(JURISDICTIONS, columns=["jurisdiction_id", "jurisdiction_name"]).drop_duplicates("jurisdiction_id")
    names[names["jurisdiction_id"].isin(scope)].to_csv(evidence / "scope_candidates.csv", index=False)
    official_totals(row, scope).to_csv(evidence / "official_totals.csv", index=False)
    evidence_files = sorted(p for p in evidence.iterdir() if p.is_file()) + [raw_target]
    case_text = (
        f"Source {city}: a new city incident feed with no prior corpus. {row['source_name']} ({row['source_url']}), "
        f"{row['portal_type']} endpoint(s) {row['api_endpoint']}. Publisher: {row['agency']} ({row['agency_type']}), "
        f"ORI {row.get('ori9') or 'n/a'}. Survey notes: years {row['first_year']}-{row['last_year']}, count unit "
        f"{row['count_unit']}, coordinates {row['coordinate_precision']}, redactions: {row['redactions'] or 'none stated'}. "
        f"The archived pull (raw_cache.parquet) holds every row the endpoint returned for {TRUTH_START}-{year_end} under "
        f"the date filter `{row['probe_where'] or '(none)'}`."
    )
    sha = common.write_packet(packet, case_text=case_text, signals={}, evidence=evidence_files)
    return packet, sha, row


def qualify(contract: dict, packet: Path, year_end: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run the contract on the archived pull: incidents with block groups, the loss funnel, a summary."""
    raw = pd.read_parquet(packet / "raw_cache.parquet")
    xw = pd.read_parquet(CROSSWALK, columns=["block_group_geoid", "jurisdiction_id"])
    empty = pd.DataFrame(columns=["incident_id", "year", "offense", "latitude", "longitude", "block_group_geoid"])
    try:
        incidents = cc.run_contract(contract, raw, year_start=TRUTH_START, year_end=year_end)
        table = cc.funnel(contract, raw, year_start=TRUTH_START, year_end=year_end, tiger_dir=TIGER, crosswalk=xw)
    except Exception as exc:  # noqa: BLE001 - a contract that cannot run does not qualify
        return empty, {"pass": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}, pd.DataFrame(columns=["offense", "year", *cc.FUNNEL_STAGES])
    incidents = cc.assign_block_groups(incidents, contract["geography"]["state_fips"], TIGER, boundary=True) if len(incidents) else empty
    total = table[table["offense"].eq("all")].iloc[0]
    in_state_share = float(total["in_state"] / total["in_years"]) if total["in_years"] else 0.0
    summary = {"raw_rows": int(total["raw"]), "mapped_rows": int(total["mapped"]), "in_years_rows": int(total["in_years"]),
               "in_state_share": in_state_share, "footprint_rows": int(total["deduplicated"]),
               "offenses_mapped": sorted(table.loc[table["year"].eq("all") & table["offense"].ne("all") & table["mapped"].gt(0), "offense"])}
    problems = []
    if not total["in_years"]:
        problems.append(f"no mapped rows in {TRUTH_START}-{year_end}")
    elif in_state_share < 0.5:
        problems.append(f"only {in_state_share:.3f} of in-year mapped rows fall in a block group of the contract's states")
    summary.update({"pass": not problems, "error": "; ".join(problems)})
    return incidents, summary, table


def _reject_by_rule(city: str, case_dir: Path, packet: Path, packet_sha: str, row: dict, drafted: tuple | None, why: str) -> dict:
    """Rule rejection of a new source (no reviewer): every offense rejected, the contract not approvable."""
    admission = {"city_key": city, "offenses": [{"offense": o, "admission_mode": "rejected", "excluded_years": [], "reason": f"rule: {why}"}
                                                for o in cc.OFFENSES_7], "rationale": "rule", "evidence_refs": []}
    how = "rule:rejected_" + ("empty_pull" if drafted is None else "deterministic_screen")
    if drafted is None:
        incidents = pd.DataFrame(columns=["incident_id", "year", "offense", "latitude", "longitude", "block_group_geoid"])
        table_qa = pd.DataFrame(columns=["year", "offense", "mapped_rows", "located_rows", "footprint_rows", "official_total", "official_source"])
        contract, draft, primary, summary, n = None, None, row["primary_jurisdiction_id"], {"pass": False, "error": why}, 0
    else:
        contract, draft, table_qa, primary, incidents, summary, exceptions = drafted
        n = len(exceptions)
    write_proposals(case_dir, city, packet, packet_sha, contract, False, (draft, None, None), admission, (None, None, None),
                    table_qa, primary, f"deterministic_screen:{why}", how)
    return _finish(city, case_dir, incidents, admission, how, False, {"mode": "new", "bg_assignment_distance": float("nan"), **summary, "pass": False},
                   why, n)


def run_city_new(city: str, run_dir: Path, year_end: int, dry_run: bool) -> dict:
    case_dir = run_dir / city
    case_dir.mkdir(parents=True, exist_ok=True)
    packet, packet_sha, row = init_new_packet(city, year_end)
    rel_packet = common.rel(packet)
    contract_schema = SKILL / "schemas" / "city_contract.json"
    tmpl = SKILL / "prompts" / "contract_new.md"
    draft_prompt = render("contract_new.md", packet_dir=rel_packet, city_key=city, year_start=str(TRUTH_START), year_end=str(year_end))
    if dry_run:
        print(f"--- {city}: packet {rel_packet} sha {packet_sha[:12]}\n{draft_prompt}")
        return {"city": city, "dry_run": True}
    add = (packet,)
    extra = f"Bash(uv run python scripts/agents/query.py --packet {rel_packet} *)"
    if pq.ParquetFile(packet / "raw_cache.parquet").metadata.num_rows == 0:
        # D screen before any agent: an endpoint that returned nothing for the window is rejected by rule.
        return _reject_by_rule(city, case_dir, packet, packet_sha, row, None, f"empty pull for {TRUTH_START}-{year_end}")
    draft = common.run_agent("drafter", draft_prompt, schema=contract_schema, out_dir=case_dir, template=tmpl, add_dirs=add,
                             extra_tools=extra, tag="contract")
    contract = dict(draft["output"], city_key=city)
    incidents, summary, table = qualify(contract, packet, year_end)
    if not summary["pass"]:
        (case_dir / "contract_under_fix.json").write_text(json.dumps(contract, indent=1) + "\n")
        (case_dir / "qualification_failed.json").write_text(json.dumps(summary, indent=1) + "\n")
        table.to_csv(case_dir / "funnel_failed.csv", index=False)
        prompt = draft_prompt + "\n\n" + render("revise_contract_new.md", city_key=city, draft_path=common.rel(case_dir / "contract_under_fix.json"),
                                                qualification_path=common.rel(case_dir / "qualification_failed.json"),
                                                funnel_path=common.rel(case_dir / "funnel_failed.csv"))
        draft = common.run_agent("drafter", prompt, schema=contract_schema, out_dir=case_dir, template=tmpl, add_dirs=add + (case_dir,),
                                 extra_tools=extra, tag="contract_fix")
        contract = dict(draft["output"], city_key=city)
        incidents, summary, table = qualify(contract, packet, year_end)

    def evidence_for(contract: dict, incidents: pd.DataFrame, summary: dict, table: pd.DataFrame):
        """Write the D evidence for this contract into the packet; return (qa table, exceptions, primary, sha)."""
        (case_dir / "qualification.json").write_text(json.dumps(summary, indent=1) + "\n")
        (packet / "evidence" / "contract.json").write_text(json.dumps(contract, indent=1) + "\n")
        table.to_csv(packet / "evidence" / "funnel.csv", index=False)
        scope = contract["geography"].get("jurisdiction_ids") or _split(row["jurisdiction_ids"])
        primary = row["primary_jurisdiction_id"] if row["primary_jurisdiction_id"] in scope else scope[0]
        official = official_totals(row, scope)
        official.to_csv(packet / "evidence" / "official_totals.csv", index=False)
        table_qa, exceptions = qa(city, contract, incidents, primary, funnel_table=table, official=official,
                                  first_year=max(TRUTH_START, int(row["first_year"] or TRUTH_START)))
        if not summary["pass"]:
            exceptions.loc[len(exceptions)] = [city, "", "", "contract_qualification", json.dumps(summary)]
        xw = pd.read_parquet(CROSSWALK, columns=["block_group_geoid", "jurisdiction_id"])
        located = incidents[incidents["block_group_geoid"].notna()].assign(block_group_geoid=lambda f: f["block_group_geoid"].astype(str).str.zfill(12))
        jur = located.merge(xw.drop_duplicates(), on="block_group_geoid", how="left").groupby("jurisdiction_id", dropna=False).size()
        names = pd.read_parquet(JURISDICTIONS, columns=["jurisdiction_id", "jurisdiction_name"]).drop_duplicates("jurisdiction_id")
        (jur.sort_values(ascending=False).rename(f"incidents_{TRUTH_START}_{year_end}").reset_index()
         .merge(names, on="jurisdiction_id", how="left").assign(in_scope=lambda f: f["jurisdiction_id"].isin(scope))
         .to_csv(packet / "evidence" / "jurisdictions.csv", index=False))
        table_qa.to_csv(packet / "evidence" / "qa_offense_year.csv", index=False)
        exceptions.to_csv(packet / "evidence" / "qa_exceptions.csv", index=False)
        (packet / "evidence" / "qualification_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        sha = common.write_packet(packet, case_text=json.loads((packet / "packet.json").read_text())["case_text"], signals={},
                                  evidence=sorted(p for p in (packet / "evidence").iterdir() if p.is_file()) + [packet / "raw_cache.parquet"])
        return table_qa, exceptions, primary, sha

    def review(tag: str, contract: dict) -> dict:
        prompt = render("review_contract_new.md", packet_dir=rel_packet, city_key=city)
        return common.run_agent("reviewer", prompt, schema=SKILL / "schemas" / "review.json", out_dir=case_dir,
                                template=SKILL / "prompts" / "review_contract_new.md", add_dirs=add, extra_tools=extra, tag=tag)

    # R after D: the reviewer sees the contract that will be used, with its funnel and QA.
    table_qa, exceptions, primary, packet_sha = evidence_for(contract, incidents, summary, table)
    if not summary["pass"] or summary["footprint_rows"] < MIN_FOOTPRINT_ROWS:
        # D screen before the reviewer: a contract that still does not qualify, or a footprint too
        # small to describe a spatial distribution, is rejected by rule.
        why = summary["error"] if not summary["pass"] else f"{summary['footprint_rows']} footprint rows < {MIN_FOOTPRINT_ROWS}"
        return _reject_by_rule(city, case_dir, packet, packet_sha, row, (contract, draft, table_qa, primary, incidents, summary, exceptions), why)
    rev = review("contract_review", contract)
    if rev["output"]["verdict"] == "revise" and rev["output"]["own_verdict"] != "accept":
        # A revise the reviewer would not sign gets one round with the objections (design 6.1, T2),
        # then a fresh review of the result. A revise it would sign keeps the contract as drafted.
        (case_dir / "contract_under_review.json").write_text(json.dumps(contract, indent=1) + "\n")
        (case_dir / "review_objections.json").write_text(json.dumps(rev["output"], indent=1) + "\n")
        prompt = draft_prompt + "\n\n" + render("revise_contract_review_new.md", city_key=city,
                                                draft_path=common.rel(case_dir / "contract_under_review.json"),
                                                review_path=common.rel(case_dir / "review_objections.json"))
        draft = common.run_agent("drafter", prompt, schema=contract_schema, out_dir=case_dir, template=SKILL / "prompts" / "revise_contract_review_new.md",
                                 add_dirs=add + (case_dir,), extra_tools=extra, tag="contract2")
        contract = dict(draft["output"], city_key=city)
        incidents, summary, table = qualify(contract, packet, year_end)
        table_qa, exceptions, primary, packet_sha = evidence_for(contract, incidents, summary, table)
        rev = review("contract_review2", contract)
    signed = rev["output"]["verdict"] == "accept" or (rev["output"]["verdict"] == "revise" and rev["output"]["own_verdict"] == "accept")
    # T3 on an unsigned contract (design 6.1): `pm_ruling.json` in the case directory, written by the
    # owner or the PM session under delegation after reading the review in full:
    # {"contract": "accept" | "reject", "held_offenses": [...], "by": ..., "note": ...}.
    ruling_path = case_dir / "pm_ruling.json"
    ruling = json.loads(ruling_path.read_text()) if ruling_path.exists() and not signed else None
    contract_ok = summary["pass"] and (signed or bool(ruling and ruling["contract"] == "accept"))

    if contract_ok:
        material = exceptions[exceptions["check"].isin(MATERIAL_CHECKS_NEW)]
        if ruling:
            # The reviewer's objections travel to the admission case as a material row.
            objections = "; ".join(o["issue"][:300] for o in rev["output"]["objections"][:4])
            material = pd.concat([material, pd.DataFrame([{"city_key": city, "offense": "", "year": "", "check": "contract_review",
                                                           "detail": f"review {rev['output']['verdict']} overruled by {ruling['by']}: "
                                                                     f"{ruling['note']}; objections: {objections}"}])], ignore_index=True)
        admission, admission_rec, how = _batched_admission(run_dir, city, case_dir, rel_packet, material, exceptions)
        if ruling:
            held = set(ruling.get("held_offenses", []))
            admission = {**admission, "offenses": [dict(o, held=f"PM ruling: {ruling['note'][:250]}") if o["offense"] in held else o
                                                   for o in admission["offenses"]]}
    else:
        # No accepted contract, no truth: a new source has no corpus to fall back on.
        reason = f"contract not accepted (qualification {'pass' if summary['pass'] else 'fail'}, review {rev['output']['verdict']})"
        # An escalated review (the evidence cannot settle the contract) holds the source for the owner.
        held = {"held": f"contract review escalated: {rev['output']['objections'][0]['issue'][:250] if rev['output']['objections'] else ''}"} \
            if rev["output"]["verdict"] == "escalate" and not ruling else {}
        if ruling:
            reason += f"; {ruling['by']}: {ruling['note'][:250]}"
        admission = {"city_key": city, "offenses": [{"offense": o, "admission_mode": "rejected", "excluded_years": [], "reason": reason, **held}
                                                    for o in cc.OFFENSES_7], "rationale": "rule", "evidence_refs": []}
        admission_rec, how = (None, None, None), "rule:rejected_contract_not_accepted"
    checks = (f"contract_qualification:{'pass' if summary['pass'] else 'fail'};in_state_share:{summary.get('in_state_share', 0):.4f};"
              f"footprint_rows:{summary.get('footprint_rows', 0)};qa_exceptions:{len(exceptions)}"
              + (f";pm_ruling:{ruling['contract']}" if ruling else ""))
    write_proposals(case_dir, city, packet, packet_sha, contract, contract_ok, (draft, rev, None), admission, admission_rec,
                    table_qa, primary, checks, how)
    if ruling:
        proposal = json.loads((case_dir / "proposal_contract.json").read_text())
        if not proposal.get("ledger_decision_id"):
            proposal.update(rule_applied=f"pm_ruling({ruling['by']}): {ruling['contract']}: {ruling['note'][:300]}",
                            evidence_refs=proposal["evidence_refs"] + ";" + common.local_ref(ruling_path))
            _write_proposal(case_dir / "proposal_contract.json", proposal)
    return _finish(city, case_dir, incidents, admission, how, contract_ok,
                   {"pass": summary["pass"], "mode": "new", "bg_assignment_distance": float("nan"), **summary},
                   "" if contract_ok else "contract not accepted", len(exceptions))


# ------------------------------------------------------------------ proposals


def _admission_rows(city: str, contract: dict | None, table: pd.DataFrame, admission: dict, primary: str, how: str) -> list[dict]:
    header = pd.read_csv(REPO_ROOT / "configs" / "city_offense_admissions.csv", nrows=0).columns.tolist()
    out = []
    by_offense = {o["offense"]: o for o in admission["offenses"]}
    for offense in cc.OFFENSES_7:
        ruling = by_offense.get(offense, {"admission_mode": "rejected", "excluded_years": [], "reason": "no ruling"})
        t = table[table["offense"].eq(offense)]
        mapped, located = int(t["mapped_rows"].sum()), int(t["located_rows"].sum())
        official = float(t["official_total"].sum()) if t["official_total"].notna().any() else np.nan
        prim = int(t["footprint_rows" if "footprint_rows" in t.columns else "primary_jurisdiction_rows"].sum())
        delta = abs(prim - official) / official if official and not np.isnan(official) else np.nan
        mode = ruling["admission_mode"]
        row = {
            "city_key": city,
            "city_name": (contract or {}).get("city_name", city),
            "jurisdiction_id": primary,
            "offense": offense,
            "admission_mode": mode,
            "share_admitted_flag": "False",
            "feed_total_candidate_flag": "False",
            "admitted_flag": str(mode == "truth_only"),
            "qa_status": "truth_only" if mode == "truth_only" else "rejected",
            "raw_offense_rows": f"{mapped:.1f}",
            "matched_share_rows": f"{located:.1f}",
            "match_share": "" if not mapped else repr(located / mapped),
            "share_sum": "1.0",
            "share_sum_pass": "True",
            "official_total": "" if np.isnan(official) else f"{official:.1f}",
            "preferred_source": ";".join(sorted(t["official_source"].dropna().astype(str).unique())),
            "quality_tier_preferred": "",
            "reconciliation_relative_delta": "" if np.isnan(delta) else repr(float(delta)),
            "reconciles_within_10pct": str(bool(not np.isnan(delta) and delta <= cc.RECONCILIATION_TOLERANCE)),
            "share_geography_pass": str(bool(mapped and located / mapped >= cc.MIN_MATCH_SHARE)),
            "total_lane_disposition": "not_admitted",
            "share_lane_disposition": "truth_only" if mode == "truth_only" else "rejected",
            "reconciliation_disposition": "truth_only_not_reconciled" if mode == "truth_only" else "rejected",
            "validation_source_name": "; ".join(s["name"] for s in (contract or {}).get("sources", [])) or f"E4 corpus {city}",
            "validation_source_url": "; ".join(s["source_url"] for s in (contract or {}).get("sources", [])),
            "adjudication": how,
            "adjudication_note": ((f"excluded_years={ruling['excluded_years']}; " if ruling.get("excluded_years") else "") + ruling.get("reason", ""))[:500],
        }
        out.append({c: row.get(c, "") for c in header})
    return out


def _agent_fields(draft: dict | None, review: dict | None, prompt: str | None) -> dict:
    if not draft:
        return {}
    return {
        "drafter_model": ";".join(draft["models"]), "drafter_effort": draft["effort"], "draft_sha256": draft["output_sha256"],
        "reviewer_model": ";".join(review["models"]) if review else "", "review_verdict": review["output"]["verdict"] if review else "",
        "review_sha256": review["output_sha256"] if review else "", "prompt_sha256": prompt or draft["prompt_sha256"],
        "cli_version": draft.get("cli_version", ""),
    }


def _write_proposal(path: Path, proposal: dict) -> None:
    """Write a proposal unless one already carries a ledger decision (approved or rejected)."""
    if path.exists() and json.loads(path.read_text()).get("ledger_decision_id"):
        return
    path.write_text(json.dumps(proposal, indent=1, sort_keys=True) + "\n")


def write_proposals(case_dir: Path, city: str, packet: Path, packet_sha: str, contract: dict | None, contract_ok: bool,
                    contract_rec: tuple, admission: dict, admission_rec: tuple, table: pd.DataFrame, primary: str,
                    checks: str, how: str) -> None:
    base = {"edition": common.EDITION, "packet_path": common.rel(packet), "packet_sha256": packet_sha,
            "evidence_refs": ";".join(common.local_ref(p) for p in sorted((packet / "evidence").iterdir()) if p.is_file()),
            "checks": checks, "affects_published": "no"}
    if contract is not None:
        payload = dict(contract)
        proposal = {**base, **_agent_fields(*contract_rec), "kind": "source_contract", "subject": f"city:{city}",
                    "target_config": f"configs/city_feeds/{city}.json", "target_key": f"city_key={city}",
                    "op": "write_json", "payload": payload, "approvable": contract_ok}
        _write_proposal(case_dir / "proposal_contract.json", proposal)
    for row in _admission_rows(city, contract, table, admission, primary, how):
        proposal = {**base, **_agent_fields(*admission_rec), "kind": "feed_admission", "subject": f"city:{city}:{row['offense']}",
                    "target_config": "configs/city_offense_admissions.csv", "target_key": f"city_key={city};offense={row['offense']}",
                    "op": "upsert_csv_unit", "key_columns": ["city_key", "offense"],
                    "key_values": {"city_key": city, "offense": row["offense"]}, "rows": [row], "approvable": True}
        if not admission_rec[0]:
            proposal["rule_applied"] = "truth_only_default_no_material_exception"
        ruling = next((o for o in admission["offenses"] if o["offense"] == row["offense"]), {})
        if ruling.get("review_verdict"):
            proposal["review_verdict"] = ruling["review_verdict"]
        held = ruling.get("held")
        if held:
            proposal["approvable"] = False
            proposal["held_for_owner"] = held
        _write_proposal(case_dir / f"proposal_admission_{row['offense']}.json", proposal)


# ------------------------------------------------------------------ per city


def run_city(city: str, run_dir: Path, year_end: int, dry_run: bool) -> dict:
    case_dir = run_dir / city
    case_dir.mkdir(parents=True, exist_ok=True)
    packet, packet_sha = init_packet(city)
    rel_packet = common.rel(packet)
    contract_schema = SKILL / "schemas" / "city_contract.json"
    review_schema = SKILL / "schemas" / "review.json"
    admission_schema = SKILL / "schemas" / "admission.json"
    draft_prompt = render("contract.md", packet_dir=rel_packet, city_key=city)
    if dry_run:
        print(f"--- {city}: packet {rel_packet} sha {packet_sha[:12]}\n{draft_prompt}")
        return {"city": city, "dry_run": True}
    add = (packet,)
    tmpl = SKILL / "prompts" / "contract.md"
    extra = f"Bash(uv run python scripts/agents/query.py --packet {rel_packet} *)"

    def review(tag: str, draft: dict) -> dict:
        prompt = render("review_contract.md", packet_dir=rel_packet, city_key=city, draft_json=json.dumps(draft["output"], indent=1))
        return common.run_agent("reviewer", prompt, schema=review_schema, out_dir=case_dir, template=SKILL / "prompts" / "review_contract.md",
                                add_dirs=add, extra_tools=extra, tag=tag)

    draft = common.run_agent("drafter", draft_prompt, schema=contract_schema, out_dir=case_dir, template=tmpl, add_dirs=add, tag="contract")
    rev = None
    if (case_dir / "contract_review.json").exists():
        # Pilot runs reviewed before the replay; their review (and revise round) is kept as recorded.
        rev = review("contract_review", draft)
        if rev["output"]["verdict"] == "revise" and (case_dir / "contract2.task.md").exists():
            prompt = draft_prompt + "\n\n" + render("../../adjudicate/prompts/revise.md", case_id=city,
                                                    draft_json=json.dumps(draft["output"], indent=1), review_json=json.dumps(rev["output"], indent=1))
            draft = common.run_agent("drafter", prompt, schema=contract_schema, out_dir=case_dir, template=tmpl, add_dirs=add, tag="contract2")
            rev = review("contract_review2", draft)
    contract = dict(draft["output"], city_key=city)
    # D before R: the replay and the live pull run first, so the drafter gets at most one fix round
    # with the deterministic diff and the reviewer sees the contract that will be used.
    counts, summary, incidents = replay(contract, packet)
    counts.to_csv(case_dir / "replay_counts.csv", index=False)
    (case_dir / "replay.json").write_text(json.dumps(summary, indent=1) + "\n")
    ext, ext_error = extend(contract, year_end, packet) if summary["pass"] else (None, "contract replay failed")
    live_mapping_error = ext_error.startswith("contract does not map") or ext_error.startswith("live pull returned")
    if not summary["pass"] or live_mapping_error:
        fails = counts[~counts["pass"]].head(40).to_dict("records")
        check = {"replay_summary": summary, "failing_cells": fails, "live_pull": ext_error or "ok"}
        prompt = draft_prompt + "\n\n" + render("revise_contract.md", city_key=city, draft_json=json.dumps(contract, indent=1),
                                                replay_json=json.dumps(check, indent=1, default=str))
        draft = common.run_agent("drafter", prompt, schema=contract_schema, out_dir=case_dir, template=tmpl, add_dirs=add, tag="contract_fix")
        contract = dict(draft["output"], city_key=city)
        counts, summary, incidents = replay(contract, packet)
        counts.to_csv(case_dir / "replay_counts_fix.csv", index=False)
        (case_dir / "replay_fix.json").write_text(json.dumps(summary, indent=1) + "\n")
        ext, ext_error = extend(contract, year_end, packet) if summary["pass"] else (None, "contract replay failed")
        rev = None
    if rev is None:
        rev = review("contract_review", draft) if not (case_dir / "contract_fix.json").exists() else review("contract_fix_review", draft)
    signed = rev["output"]["verdict"] == "accept" or (rev["output"]["verdict"] == "revise" and rev["output"]["own_verdict"] == "accept")
    contract_ok = summary["pass"] and signed
    if not contract_ok:
        ext, ext_error = None, ext_error or "contract not accepted"

    # Truth rows 2018-2024 are the frozen E4 corpus itself; the contract only extends it.
    base = oracle(city)
    parts = [base]
    if ext is not None and not ext.empty:
        ext.to_parquet(case_dir / f"extension_{year_end}.parquet", index=False)
        parts.append(ext)
    combined = pd.concat(parts, ignore_index=True)
    jur = pd.read_csv(packet / "evidence" / "jurisdictions.csv")
    scope = contract["geography"]["jurisdiction_ids"] if contract.get("geography", {}).get("jurisdiction_ids") else []
    primary = next((j for j in jur["jurisdiction_id"] if j in scope), jur["jurisdiction_id"].iloc[0])
    table, exceptions = qa(city, contract, combined, primary)
    if not summary["pass"]:
        exceptions.loc[len(exceptions)] = [city, "", "", "contract_replay", json.dumps(summary)]
    if ext_error:
        exceptions.loc[len(exceptions)] = [city, "", year_end, "extension", ext_error]
    table.to_csv(packet / "evidence" / "qa_offense_year.csv", index=False)
    exceptions.to_csv(packet / "evidence" / "qa_exceptions.csv", index=False)
    (packet / "evidence" / "replay_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    packet_sha = common.write_packet(packet, case_text=json.loads((packet / "packet.json").read_text())["case_text"], signals={},
                                     evidence=sorted(p for p in (packet / "evidence").iterdir() if p.is_file()) + [packet / "raw_cache.parquet"])

    material = exceptions[exceptions["check"].isin(MATERIAL_CHECKS)
                          & ~(exceptions["check"].eq("year_coverage") & exceptions["detail"].str.contains(f"\\[{year_end}\\]"))]
    admission, admission_rec, how = _batched_admission(run_dir, city, case_dir, rel_packet, material, exceptions)
    checks = f"contract_replay:{'pass' if summary['pass'] else 'fail'};bg_distance:{summary['bg_assignment_distance']:.4f};qa_exceptions:{len(exceptions)}"
    write_proposals(case_dir, city, packet, packet_sha, contract, contract_ok, (draft, rev, None), admission, admission_rec,
                    table, primary, checks, how)
    return _finish(city, case_dir, combined, admission, how, contract_ok, summary, ext_error, len(exceptions))


def _batched_admission(run_dir: Path, city: str, case_dir: Path, rel_packet: str, material: pd.DataFrame,
                       exceptions: pd.DataFrame) -> tuple[dict, tuple, str]:
    """The truth_only rule when QA has no material exception, else this city's ruling from the run's
    batched drafter/reviewer case (NeedsBatch until that case exists)."""
    if material.empty:
        admission = {"city_key": city, "offenses": [{"offense": o, "admission_mode": "truth_only", "excluded_years": [],
                                                      "reason": "no material QA exception" + ("" if exceptions.empty else f"; recorded: {sorted(set(exceptions['check']))}")}
                                                     for o in cc.OFFENSES_7], "rationale": "rule", "evidence_refs": []}
        return admission, (None, None, None), "rule:truth_only_default_no_material_exception"
    batch = run_dir / "_admission_batch"
    chunks = json.loads((batch / "chunks.json").read_text()) if (batch / "chunks.json").exists() else {}
    suffix = f"_{chunks[city]}" if city in chunks else ""
    rec = batch / f"admission_review{suffix}.json"
    if not rec.exists():
        (case_dir / "needs_admission.json").write_text(json.dumps({"packet": rel_packet, "material": material.to_dict("records")}, indent=1, default=str) + "\n")
        raise NeedsBatch(city)
    adm = json.loads((batch / f"admission{suffix}.json").read_text())
    adm_rev = json.loads(rec.read_text())
    ruling = next((c for c in adm["output"]["cities"] if c["city_key"] == city), None)
    if ruling is None:
        raise NeedsBatch(city)
    # Per-offense disposition from the reviewer's field verdicts (`<city>.<offense>`): an offense
    # the reviewer does not accept is held for the owner (no approvable proposal, not in the
    # truth surface) rather than re-drafted.
    fields = {f["field"]: f for f in adm_rev["output"]["field_verdicts"]}
    offenses = []
    for o in ruling["offenses"]:
        f = fields.get(f"{city}.{o['offense']}")
        verdict = f["verdict"] if f else adm_rev["output"]["verdict"]
        o = dict(o, review_verdict=verdict)
        offenses.append(o if verdict == "accept" else dict(o, held=f"review {verdict}: {(f or {}).get('note', '')[:300]}"))
    return {**ruling, "offenses": offenses}, (adm, adm_rev, None), f"drafter+reviewer(batch):{adm_rev['output']['verdict']}"


def _finish(city: str, case_dir: Path, combined: pd.DataFrame, admission: dict, how: str, contract_ok: bool, summary: dict,
            ext_error: str, n_exceptions: int) -> dict:
    truth = cc.truth_rows(combined, city_key=city, source_kind="cad_dispatch" if city in CAD_SOURCES else "incident_report",
                          crosswalk=pd.read_parquet(CROSSWALK, columns=["block_group_geoid", "jurisdiction_id"]))
    excluded = {(o["offense"], y) for o in admission["offenses"] for y in o.get("excluded_years", [])}
    admitted = {o["offense"] for o in admission["offenses"] if o["admission_mode"] == "truth_only" and not o.get("held")}
    held = {o["offense"]: o["held"] for o in admission["offenses"] if o.get("held")}
    truth = truth[truth["offense"].isin(admitted) & ~truth[["offense", "year"]].apply(tuple, axis=1).isin(excluded)]
    truth.to_parquet(case_dir / "truth_rows.parquet", index=False)
    result = {"city": city, "contract_ok": contract_ok, "replay": summary, "extension_error": ext_error,
              "qa_exceptions": int(n_exceptions), "admitted_offenses": sorted(admitted),
              "rejected_offenses": sorted(set(cc.OFFENSES_7) - admitted - set(held)), "held_offenses": held, "adjudication": how,
              "years": sorted(int(y) for y in truth["year"].unique()), "truth_incidents": int(truth["incident_count"].sum())}
    (case_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


ADMISSION_CHUNK = 8  # cities per batched admission case in a new-source run


def run_admission_batch(run_dir: Path, cities: list[str], new: bool = False) -> None:
    """One drafter and one reviewer call for the run's cities with material QA exceptions (E4), or
    per chunk of ADMISSION_CHUNK cities (new sources; the chunking is fixed in `chunks.json`)."""
    batch = run_dir / "_admission_batch"
    batch.mkdir(parents=True, exist_ok=True)
    groups: list[tuple[str, list[str]]] = [("", sorted(cities))]
    if new:
        chunks_path = batch / "chunks.json"
        chunks = json.loads(chunks_path.read_text()) if chunks_path.exists() else {}
        fresh = [c for c in sorted(cities) if c not in chunks]
        start = max(chunks.values(), default=0)
        for i in range(0, len(fresh), ADMISSION_CHUNK):
            for c in fresh[i:i + ADMISSION_CHUNK]:
                chunks[c] = start + 1 + i // ADMISSION_CHUNK
        chunks_path.write_text(json.dumps(chunks, indent=1, sort_keys=True) + "\n")
        groups = [(f"_{k}", sorted(c for c in cities if chunks[c] == k)) for k in sorted({chunks[c] for c in cities})]
    pending = None
    for suffix, members in groups:
        try:
            _admission_case(run_dir, batch, members, new, suffix)
        except common.PendingTask as task:
            pending = pending or task
    if pending:
        raise pending


def _admission_case(run_dir: Path, batch: Path, cities: list[str], new: bool, tag_suffix: str) -> None:
    needs = {c: json.loads((run_dir / c / "needs_admission.json").read_text()) for c in cities}
    city_list = "\n".join(f"- {c}: {n['packet']}" for c, n in sorted(needs.items()))
    material = "\n".join(f"- {c}: {m['check']}, {m['offense']}, {m['year']}, {m['detail']}" for c, n in sorted(needs.items()) for m in n["material"])
    suffix = "_new" if new else ""
    material_path = batch / f"material{tag_suffix}.csv"
    pd.DataFrame([{"city_key": c, **m} for c, n in sorted(needs.items()) for m in n["material"]]).to_csv(material_path, index=False)
    prompt = render(f"feed_exception_batch{suffix}.md", city_list=city_list, material=material, material_path=common.rel(material_path))
    packets = tuple(PACKETS / c for c in sorted(needs))
    adm = common.run_agent("drafter", prompt, schema=SKILL / "schemas" / "admission_batch.json", out_dir=batch,
                           template=SKILL / "prompts" / f"feed_exception_batch{suffix}.md", add_dirs=packets + (batch,), tag=f"admission{tag_suffix}")
    draft_path = batch / f"admission_draft{tag_suffix}.json"
    draft_path.write_text(json.dumps(adm["output"], indent=1) + "\n")
    review_prompt = render(f"review_feed_exception_batch{suffix}.md", city_list=city_list, draft_json=json.dumps(adm["output"], indent=1),
                           draft_path=common.rel(draft_path))
    common.run_agent("reviewer", review_prompt, schema=SKILL / "schemas" / "review.json", out_dir=batch,
                     template=SKILL / "prompts" / f"review_feed_exception_batch{suffix}.md", add_dirs=packets + (batch,),
                     tag=f"admission_review{tag_suffix}")


# ------------------------------------------------------------------ share admission (stage 7)

ADMISSIONS = REPO_ROOT / "configs" / "city_offense_admissions.csv"
SOURCES = REPO_ROOT / "configs" / "city_incident_sources.csv"
PRIORITY = REPO_ROOT / "configs" / "city_incident_priority.csv"
YEAR_EXCLUSIONS = REPO_ROOT / "configs" / "city_feed_year_exclusions.csv"
SHARE_DIR = "share_admission"


def _share_row(city: str, offense: str, ruling: dict, qa: pd.DataFrame, existing: dict, meta: dict, how: str) -> dict:
    """A `configs/city_offense_admissions.csv` row for a share-admission ruling, from the city's
    `share_admission/qa.csv` over the years the ruling keeps (`existing` fills the other fields)."""
    header = pd.read_csv(ADMISSIONS, nrows=0).columns.tolist()
    mode, excluded = ruling["admission_mode"], [int(y) for y in ruling.get("excluded_years", [])]
    t = qa[qa["offense"].eq(offense) & ~qa["year"].isin(excluded)]
    mapped, located = float(t["mapped_rows"].sum()), float(t["located_rows"].sum())
    official = float(t["official_total"].sum()) if t["official_total"].notna().any() else np.nan
    delta = abs(located - official) / official if official and not np.isnan(official) else np.nan
    share, within = mode in cc.SHARE_MODES, bool(not np.isnan(delta) and delta <= cc.RECONCILIATION_TOLERANCE)
    row = {c: existing.get(c, "") for c in header}
    row.update({
        "city_key": city, "offense": offense, "city_name": meta["city_name"], "jurisdiction_id": meta["jurisdiction_id"],
        "admission_mode": mode, "share_admitted_flag": str(share), "feed_total_candidate_flag": "False",
        "admitted_flag": str(mode != "rejected"),
        "qa_status": {"share_only": "pass_share_only", "truth_only": "truth_only"}.get(mode, "rejected"),
        "raw_offense_rows": f"{mapped:.1f}", "matched_share_rows": f"{located:.1f}",
        "match_share": "" if not mapped else repr(located / mapped), "share_sum": "1.0", "share_sum_pass": "True",
        "official_total": "" if np.isnan(official) else f"{official:.1f}",
        "preferred_source": ";".join(sorted(t["official_source"].dropna().astype(str).unique())),
        "reconciliation_relative_delta": "" if np.isnan(delta) else repr(float(delta)), "reconciles_within_10pct": str(within),
        "share_geography_pass": str(bool(mapped and located / mapped >= cc.MIN_MATCH_SHARE)),
        "total_lane_disposition": "official_control" if share else "not_admitted",
        "share_lane_disposition": "admit_share_only" if share else ("truth_only" if mode == "truth_only" else "rejected"),
        "reconciliation_disposition": ("reconciled_within_strict_10pct" if within else "unreconciled_keep_share_only") if share
        else ("truth_only_not_reconciled" if mode == "truth_only" else "rejected"),
        "validation_source_name": meta["source_name"], "validation_source_url": meta["source_url"], "adjudication": how,
        "adjudication_note": ((f"excluded_years={excluded}; " if excluded else "") + ruling.get("reason", ""))[:500],
    })
    return row


def _share_meta(city: str) -> dict:
    sources = pd.read_csv(SOURCES, dtype=str).fillna("")
    contract_path = cc.contract_dir(REPO_ROOT) / f"{city}.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        registry = registry_row(city) if (PACKETS / city / "evidence" / "registry_row.json").exists() else {}
        return {"city_name": contract["city_name"], "contract": contract, "registry": registry,
                "jurisdiction_id": registry.get("primary_jurisdiction_id") or contract["geography"]["jurisdiction_ids"][0],
                "source_name": "; ".join(s["name"] for s in contract["sources"]),
                "source_url": "; ".join(s["source_url"] for s in contract["sources"])}
    row = sources[sources["city_key"].eq(city)].iloc[0]
    return {"city_name": row["city_name"], "contract": None, "registry": {}, "jurisdiction_id": row["jurisdiction_id"],
            "source_name": row["source_name"], "source_url": row["source_url"]}


def _sources_row(city: str, meta: dict, share_offenses: list[str], years: list[int]) -> dict:
    """The governed `configs/city_incident_sources.csv` row of a share-admitted contract feed."""
    header = pd.read_csv(SOURCES, nrows=0).columns.tolist()
    sources = pd.read_csv(SOURCES, dtype=str).fillna("")
    existing = sources[sources["city_key"].eq(city)]
    row = existing.iloc[0].to_dict() if len(existing) else {c: "" for c in header}
    contract, registry = meta["contract"], meta["registry"]
    mapping = contract["mapping"]
    row.update({
        "city_key": city, "jurisdiction_id": meta["jurisdiction_id"], "city_name": contract["city_name"],
        "state_abbr": registry.get("state_abbr", row.get("state_abbr", "")), "source_name": meta["source_name"],
        "source_url": " ; ".join(s["api_endpoint"] for s in contract["sources"]),
        "portal_type": ";".join(sorted({s["portal_type"] for s in contract["sources"]})),
        "coverage_start_year": str(min(years)), "coverage_end_year": str(max(years)), "years_usable": str(len(years)),
        "offense_fields_present": ",".join(sorted({c["field"] for r in mapping["offense_rules"] for c in r.get("conditions", [])})),
        "date_field": mapping.get("year_field") or mapping["date_field"],
        "location_fields_present": f"{mapping['latitude_field']},{mapping['longitude_field']}", "latlon_present": "yes",
        "address_present": row.get("address_present") or "unknown", "block_group_join_ready": "yes",
        "geocode_quality_tier": row.get("geocode_quality_tier") or "contract_reported_latlon",
        "dedupe_key_available": "yes" if mapping.get("dedupe_on_id", True) else "no",
        "offense_crosswalk_complexity": row.get("offense_crosswalk_complexity") or "contract", "recommended_disposition": "ready_now",
        "analyst_notes": f"Contract feed configs/city_feeds/{city}.json (stage 7 contract builder); share_only for "
                         f"{', '.join(share_offenses)} by run-share admission; other offenses not in the share surface.",
    })
    return {c: row.get(c, "") for c in header}


def run_share_admission(run_id: str, cities: list[str]) -> list[Path]:
    """One drafter and one reviewer call ruling `share_only` / `truth_only` / `rejected` per city and
    offense from each city's `share_admission/` evidence; writes the proposals for `ledger.py approve`."""
    run_dir = RUNS_DIR / run_id
    batch = run_dir / "_share_admission"
    batch.mkdir(parents=True, exist_ok=True)
    shas = {}
    for city in cities:
        share_dir = PACKETS / city / SHARE_DIR
        if not (share_dir / "rule.md").exists() or not (share_dir / "qa.csv").exists():
            raise SystemExit(f"{common.rel(share_dir)} needs rule.md and qa.csv")
        shas[city] = common.write_packet(share_dir, case_text=f"Share admission for {city}: see rule.md.", signals={},
                                         evidence=sorted(p for p in share_dir.iterdir() if p.is_file() and p.name != "packet.json"))
    city_list = "\n".join(f"- {c}: {common.rel(PACKETS / c)}" for c in sorted(cities))
    packets = tuple(PACKETS / c for c in sorted(cities))
    adm = common.run_agent("drafter", render("share_admission_batch.md", city_list=city_list),
                           schema=SKILL / "schemas" / "share_admission_batch.json", out_dir=batch,
                           template=SKILL / "prompts" / "share_admission_batch.md", add_dirs=packets, tag="share_admission")
    draft_path = batch / "share_admission_draft.json"
    draft_path.write_text(json.dumps(adm["output"], indent=1) + "\n")
    rev = common.run_agent("reviewer", render("review_share_admission_batch.md", city_list=city_list, draft_path=common.rel(draft_path)),
                           schema=SKILL / "schemas" / "review.json", out_dir=batch,
                           template=SKILL / "prompts" / "review_share_admission_batch.md", add_dirs=packets + (batch,),
                           tag="share_admission_review")
    fields = {f["field"]: f for f in rev["output"]["field_verdicts"]}
    admissions = pd.read_csv(ADMISSIONS, dtype=str).fillna("")
    how = f"drafter+reviewer(share batch):{rev['output']['verdict']}"
    written = []
    for city in sorted(cities):
        ruling = next((c for c in adm["output"]["cities"] if c["city_key"] == city), None)
        if ruling is None:
            raise common.AgentError(f"share admission draft has no entry for {city}")
        meta = _share_meta(city)
        qa = pd.read_csv(PACKETS / city / SHARE_DIR / "qa.csv")
        share_dir = PACKETS / city / SHARE_DIR
        base = {"edition": common.EDITION, "packet_path": common.rel(share_dir), "packet_sha256": shas[city],
                "evidence_refs": ";".join(common.local_ref(p) for p in sorted(share_dir.iterdir()) if p.is_file()),
                "checks": f"share_admission:{rev['output']['verdict']}", "affects_published": "yes",
                **_agent_fields(adm, rev, None)}
        case_dir = run_dir / city
        case_dir.mkdir(parents=True, exist_ok=True)
        approvable_share, years = [], set()
        for o in ruling["offenses"]:
            f = fields.get(f"{city}.{o['offense']}")
            verdict = f["verdict"] if f else rev["output"]["verdict"]
            existing = admissions[admissions["city_key"].eq(city) & admissions["offense"].eq(o["offense"])]
            row = _share_row(city, o["offense"], o, qa, existing.iloc[0].to_dict() if len(existing) else {}, meta, how)
            proposal = {**base, "kind": "feed_admission", "subject": f"city:{city}:{o['offense']}",
                        "target_config": common.rel(ADMISSIONS), "target_key": f"city_key={city};offense={o['offense']}",
                        "op": "upsert_csv_unit", "key_columns": ["city_key", "offense"],
                        "key_values": {"city_key": city, "offense": o["offense"]}, "rows": [row],
                        "approvable": verdict == "accept", "review_verdict": verdict,
                        "excluded_years": [int(y) for y in o.get("excluded_years", [])]}
            if verdict != "accept":
                proposal["held_for_owner"] = f"review {verdict}: {(f or {}).get('note', '')[:300]}"
            elif o["admission_mode"] in cc.SHARE_MODES:
                approvable_share.append(o["offense"])
                kept = qa[qa["offense"].eq(o["offense"]) & ~qa["year"].isin(o.get("excluded_years", []))]
                years |= set(int(y) for y in kept.loc[kept["located_rows"].gt(0), "year"])
            path = case_dir / f"proposal_share_{o['offense']}.json"
            _write_proposal(path, proposal)
            written.append(path)
        if approvable_share and meta["contract"] is not None:
            row = _sources_row(city, meta, sorted(approvable_share), sorted(years))
            proposal = {**base, "kind": "source_contract", "subject": f"city:{city}", "target_config": common.rel(SOURCES),
                        "target_key": f"city_key={city}", "op": "upsert_csv_unit", "key_columns": ["city_key"],
                        "key_values": {"city_key": city}, "rows": [row], "approvable": True,
                        "rule_applied": "sources row of a share-admitted contract feed (deterministic from the contract)"}
            path = case_dir / "proposal_share_sources.json"
            _write_proposal(path, proposal)
            written.append(path)
    return written


def wire_share_feeds(run_id: str, cities: list[str]) -> list[str]:
    """After approval: the ungoverned stage-7 wiring that follows the approved share admissions --
    the packet gate files, the priority row of a contract feed and the year exclusions."""
    run_dir = RUNS_DIR / run_id
    admissions = pd.read_csv(ADMISSIONS, dtype=str).fillna("")
    exclusions = pd.read_csv(YEAR_EXCLUSIONS, dtype=str).fillna("")
    priority = pd.read_csv(PRIORITY, dtype=str).fillna("")
    changed = []
    for city in sorted(cities):
        proposals = [json.loads(p.read_text()) for p in sorted((run_dir / city).glob("proposal_share_*.json"))]
        rows = [p for p in proposals if p["kind"] == "feed_admission"]
        if not rows or any(p["approvable"] and not p.get("ledger_decision_id") for p in proposals):
            print(f"{city}: share proposals not all decided; nothing wired")
            continue
        share = admissions[admissions["city_key"].eq(city) & admissions["admission_mode"].isin(cc.SHARE_MODES)]
        if share.empty:
            # No share offense: the city is not wired (a release city keeps its own loader and gate).
            print(f"{city}: no share-mode admission; nothing wired")
            continue
        meta = _share_meta(city)
        packet = PACKETS / city
        status_path, gate_path = packet / "packet_status.csv", packet / "packet_offense_status.csv"
        decided = {p["rows"][0]["offense"]: p for p in rows if p.get("ledger_decision_id")}
        if gate_path.exists():
            gate = pd.read_csv(gate_path, dtype=str).fillna("")
        else:
            gate = pd.DataFrame({"offense": list(cc.OFFENSES_7), "production_ready": "no", "city_share_integration_status": "blocked",
                                 "comparison_class": "", "notes": ""})
        for offense, p in decided.items():
            on = offense in set(share["offense"])
            note = f"{p['ledger_decision_id']} ({run_id}): {p['rows'][0]['adjudication_note'][:300]}"
            mask = gate["offense"].eq(offense)
            if not mask.any():
                gate.loc[len(gate), "offense"] = offense
                mask = gate["offense"].eq(offense)
            gate.loc[mask, ["production_ready", "city_share_integration_status", "comparison_class", "notes"]] = [
                "yes" if on else "no", "offense_selective" if on else "blocked", "share_admission_2025_3", note]
        gate.to_csv(gate_path, index=False)
        changed.append(common.rel(gate_path))
        if not status_path.exists() and len(share):
            registry = meta["registry"]
            pd.DataFrame([{"city_key": city, "city_name": meta["city_name"], "jurisdiction_id": meta["jurisdiction_id"],
                           "state_abbr": registry.get("state_abbr", ""), "priority_bucket": "contract_feed", "priority_rank": "",
                           "packet_status": "share_admitted", "current_owner": "onboard_feed.py --share-admission",
                           "production_ready": "yes", "city_share_integration_status": "offense_selective",
                           "reconciliation_status": f"see {common.rel(packet / SHARE_DIR)}",
                           "notes": f"Contract feed; share_only offenses: {', '.join(sorted(share['offense']))} ({run_id})."}]
                         ).to_csv(status_path, index=False)
            changed.append(common.rel(status_path))
        if len(share) and meta["contract"] is not None and not priority["city_key"].eq(city).any():
            rank = int(pd.to_numeric(priority["priority_rank"], errors="coerce").max()) + 1
            priority.loc[len(priority)] = {"city_key": city, "jurisdiction_id": meta["jurisdiction_id"], "city_name": meta["city_name"],
                                           "state_abbr": meta["registry"].get("state_abbr", ""), "priority_bucket": "contract_feed",
                                           "priority_rank": str(rank), "basis": f"Share-admitted contract feed ({run_id}).",
                                           "notes": f"share_only: {', '.join(sorted(share['offense']))}"}
            if status_path.exists():
                status = pd.read_csv(status_path, dtype=str).fillna("")
                status["priority_rank"] = str(rank)
                status.to_csv(status_path, index=False)
            changed.append(common.rel(PRIORITY))
        for offense, p in decided.items():
            if offense not in set(share["offense"]):
                continue
            years = p["excluded_years"]
            for year in years:
                key = exclusions["city_key"].eq(city) & exclusions["offense"].eq(offense) & exclusions["year"].eq(str(year))
                reason = f"share admission {p['ledger_decision_id']} ({run_id}) excludes this year: {p['rows'][0]['adjudication_note'][:240]}"
                if key.any():
                    exclusions.loc[key, "reason"] = reason
                else:
                    exclusions.loc[len(exclusions)] = {"city_key": city, "offense": offense, "year": str(year), "reason": reason}
            if years:
                changed.append(common.rel(YEAR_EXCLUSIONS))
    priority.to_csv(PRIORITY, index=False)
    exclusions.to_csv(YEAR_EXCLUSIONS, index=False)
    return sorted(set(changed))


def approval_summary(run_id: str, results: list[dict]) -> str:
    lines = [f"# Approval summary: {run_id}", "", "| city | contract | replay or qualification | bg distance | extension | QA exceptions | admitted | rejected | held for owner | adjudication |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["city"]):
        lines.append(f"| {r['city']} | {'accepted' if r['contract_ok'] else 'not accepted'} | {'pass' if r['replay']['pass'] else 'fail'} | "
                     f"{r['replay']['bg_assignment_distance']:.4f} | {r['extension_error'] or 'ok'} | {r['qa_exceptions']} | "
                     f"{len(r['admitted_offenses'])} | {', '.join(r['rejected_offenses']) or '-'} | {', '.join(r.get('held_offenses', {})) or '-'} | {r['adjudication']} |")
    sample = random.Random(20260924).sample(sorted(r["city"] for r in results), min(20, len(results)))
    lines += ["", "No unit changes a published value (truth_only admissions and contracts are not read by stages 1-12).",
              "", f"Seeded sample to read in full (seed 20260924): {', '.join(sorted(sample))}"]
    return "\n".join(lines) + "\n"


def sealed_sources() -> set[str]:
    if not SEALS.exists():
        return set()
    common.core.require_approved(SEALS)
    return set(pd.read_csv(SEALS, dtype=str, keep_default_na=False)["city_key"])


def assemble_truth_surface() -> tuple[int, int]:
    """Union the truth rows of every city whose admissions are approved in the ledger.

    Writes state/modeling/e4_truth_surface_2018_2025.parquet (the widened-surface layout plus
    `admission_decision_id`) and the combined per-source QA table next to it. Only offenses with an
    effective, approved `truth_only` ledger row enter. Sources in `configs/truth_source_seals.csv`
    go to `truth_surface_sealed_2018_2025.parquet` instead and never to the training surface.
    """
    ledger = common.read_ledger()
    eff = common.core.effective_rows(ledger)
    adm = eff[eff["target_config"].eq("configs/city_offense_admissions.csv") & eff["human_approval"].eq("approved")]
    rows = pd.read_csv(REPO_ROOT / "configs" / "city_offense_admissions.csv", dtype=str).fillna("")
    truth_only = rows[rows["admission_mode"].eq("truth_only")]
    approved = {}
    for key, decision in zip(adm["target_key"], adm["decision_id"]):
        parts = dict(p.split("=", 1) for p in key.split(";"))
        if ((truth_only["city_key"] == parts["city_key"]) & (truth_only["offense"] == parts["offense"])).any():
            approved[(parts["city_key"], parts["offense"])] = decision
    frames, qa_frames = [], []
    for result in sorted(RUNS_DIR.glob("*/*/result.json")):
        city = result.parent.name
        if not any(c == city for c, _ in approved) or not (result.parent / "truth_rows.parquet").exists():
            continue
        truth = pd.read_parquet(result.parent / "truth_rows.parquet")
        truth["admission_decision_id"] = [approved.get((city, o), "") for o in truth["offense"]]
        frames.append(truth[truth["admission_decision_id"].ne("")])
        qa_frames.append(pd.read_csv(PACKETS / city / "evidence" / "qa_offense_year.csv"))
    out_dir = REPO_ROOT / "state" / "modeling"
    if not frames:
        return 0, 0
    surface = pd.concat(frames, ignore_index=True).drop_duplicates(["source", "year", "offense", "block_group_geoid", "jurisdiction_id"], keep="last")
    surface = surface.sort_values(["source", "year", "offense", "block_group_geoid", "jurisdiction_id"]).reset_index(drop=True)
    sealed = surface["source"].isin(sealed_sources())
    surface[~sealed].reset_index(drop=True).to_parquet(out_dir / "e4_truth_surface_2018_2025.parquet", index=False)
    surface[sealed].reset_index(drop=True).to_parquet(out_dir / "truth_surface_sealed_2018_2025.parquet", index=False)
    pd.concat(qa_frames, ignore_index=True).to_csv(out_dir / "e4_truth_surface_qa.csv", index=False)
    return int(surface.loc[~sealed, "source"].nunique()), int(surface.loc[~sealed, "incident_count"].sum())


def seal_draw(cities: list[dict], seed: int) -> list[str]:
    """One third of `cities` ({city_key, state_fips, population}), stratified: sources are ordered by
    stratum (Census region, population class), shuffled within each stratum by the seeded RNG, and
    every third one from a seeded start is sealed."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[str]] = {}
    for c in cities:
        size = next(label for floor, label in SIZE_CLASSES if float(c["population"] or 0) >= floor)
        strata.setdefault((CENSUS_REGION[str(c["state_fips"]).zfill(2)], size), []).append(c["city_key"])
    ordered = []
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        ordered += members
    return sorted(ordered[rng.randrange(3)::3])


def propose_seals(run_id: str, seed: int) -> list[str]:
    """`evaluation_seal` proposals for one third of the run's new sources with approved truth_only rows."""
    run_dir = RUNS_DIR / run_id
    ledger = common.core.effective_rows(common.read_ledger())
    adm = ledger[ledger["target_config"].eq("configs/city_offense_admissions.csv") & ledger["human_approval"].eq("approved")]
    admitted = {dict(p.split("=", 1) for p in k.split(";"))["city_key"] for k in adm["target_key"]}
    rows = pd.read_csv(REPO_ROOT / "configs" / "city_offense_admissions.csv", dtype=str).fillna("")
    admitted &= set(rows.loc[rows["admission_mode"].eq("truth_only"), "city_key"])
    eligible = []
    for result in sorted(run_dir.glob("*/result.json")):
        city = result.parent.name
        registry = PACKETS / city / "evidence" / "registry_row.json"
        if city in admitted and registry.exists():
            row = json.loads(registry.read_text())
            eligible.append({"city_key": city, "state_fips": row["state_fips"], "population": row["population"]})
    sealed = seal_draw(eligible, seed)
    out = run_dir / "_seal"
    out.mkdir(parents=True, exist_ok=True)
    draw = out / "draw.json"
    draw.write_text(json.dumps({"seed": seed, "eligible": eligible, "sealed": sealed, "size_classes": SIZE_CLASSES}, indent=1) + "\n")
    for city in sealed:
        c = next(e for e in eligible if e["city_key"] == city)
        size = next(label for floor, label in SIZE_CLASSES if float(c["population"] or 0) >= floor)
        row = {"city_key": city, "sealed_in_run": run_id, "seed": str(seed), "stratum": f"{CENSUS_REGION[str(c['state_fips']).zfill(2)]}/{size}",
               "rule": "never trained on; scored once at the end of the edition"}
        proposal = {"edition": common.EDITION, "kind": "evaluation_seal", "subject": f"city:{city}",
                    "target_config": common.rel(SEALS), "target_key": f"city_key={city}", "op": "upsert_csv_unit",
                    "key_columns": ["city_key"], "key_values": {"city_key": city}, "rows": [row], "approvable": True,
                    "packet_path": common.rel(draw), "packet_sha256": common.sha256_file(draw), "evidence_refs": common.local_ref(draw),
                    "rule_applied": f"seeded_stratified_third(seed={seed})", "checks": f"eligible:{len(eligible)};sealed:{len(sealed)}",
                    "affects_published": "no"}
        _write_proposal(out / f"proposal_seal_{city}.json", proposal)
    return sealed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", action="append")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-e4", action="store_true", help="replay an E4 corpus city against its archived oracle")
    source.add_argument("--new", action="store_true", help="qualify a new source from configs/city_source_registry.csv")
    parser.add_argument("--seal", type=int, metavar="SEED", help="propose evaluation seals for the run's admitted new sources")
    parser.add_argument("--share-admission", action="store_true",
                        help="rule share_only per offense from share_admission/ evidence (drafter, reviewer); --wire after approval")
    parser.add_argument("--wire", action="store_true", help="with --share-admission: write the stage-7 wiring of approved share admissions")
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=["headless", "session"], default="headless")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seal is not None:
        sealed = propose_seals(args.run, args.seal)
        print(f"sealed {len(sealed)}: {sealed}")
        print(f"pending for the owner: uv run python scripts/agents/ledger.py approve --run {args.run} --cases _seal --approved-by <who>")
        return 0
    if args.share_admission:
        common.set_mode(args.mode)
        cities = [c for arg in args.city or [] for c in arg.split(",") if c]
        if args.wire:
            print("\n".join(wire_share_feeds(args.run, cities)) or "nothing wired")
            return 0
        try:
            written = run_share_admission(args.run, cities)
        except common.PendingTask as task:
            print(f"  TASK {task}")
            return 2
        print(f"{len(written)} proposals; pending for the owner: uv run python scripts/agents/ledger.py approve --run {args.run} "
              f"--cases {','.join(sorted(cities))} --approved-by <who>")
        return 0
    if not args.city or not (args.from_e4 or args.new):
        parser.error("--city with --from-e4 or --new is required (or --seal SEED)")
    common.set_mode(args.mode)
    run_dir = RUNS_DIR / args.run
    run_one = run_city_new if args.new else run_city
    cities = [c for arg in args.city for c in arg.split(",") if c]
    results, pending, failures, batch = [], {}, {}, []
    for city in cities:
        try:
            results.append(run_one(city, run_dir, args.year_end, args.dry_run))
        except common.PendingTask as task:
            pending[city] = str(task)
        except common.AgentError as exc:
            failures[city] = str(exc)
        except NeedsBatch:
            batch.append(city)
    if batch and not pending:
        try:
            run_admission_batch(run_dir, batch, new=args.new)
            for city in batch:
                results.append(run_one(city, run_dir, args.year_end, args.dry_run))
        except common.PendingTask as task:
            pending["_admission_batch"] = str(task)
        except common.AgentError as exc:
            failures["_admission_batch"] = str(exc)
    elif batch:
        print(f"waiting for the other cities before the batched admission case: {batch}")
    if args.dry_run:
        return 0
    state = common.load_state(run_dir)
    state.update(cities=cities, results={r["city"]: r for r in results}, pending_tasks=pending, failures=failures)
    common.save_state(run_dir, state)
    all_results = list(state["results"].values())
    (run_dir / "approval_summary.md").write_text(approval_summary(args.run, all_results))
    for r in results:
        print(f"{r['city']}: contract {'accepted' if r['contract_ok'] else 'NOT accepted'}, replay {r['replay']}, "
              f"extension {r['extension_error'] or 'ok'}, QA exceptions {r['qa_exceptions']}, admitted {r['admitted_offenses']}")
    for city, task in pending.items():
        print(f"  TASK {task}")
    for city, err in failures.items():
        print(f"  FAILED {city}: {err[:300]}")
    sources, incidents = assemble_truth_surface()
    print(f"truth surface: {sources} approved sources, {incidents} incidents -> state/modeling/e4_truth_surface_2018_2025.parquet")
    print(f"pending for the owner: uv run python scripts/agents/ledger.py summary --run {args.run}")
    return 2 if pending else (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
