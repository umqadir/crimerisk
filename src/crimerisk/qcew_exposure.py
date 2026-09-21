"""QCEW exposure updating: the LODES *pattern* kept, its *level* moved to the target year.

PLAN.md item 6, verbatim: "EXPOSURE UPDATING: LODES fine pattern scaled by QCEW county-industry
totals to target year; denominator provenance/vintage published; exposure lane renamed a 'modeled
place-exposure proxy.'"

The v2 exposure lane (`exposure_ensemble.py`) carries two LODES-derived legs:

* the **daytime-jobs leg**, `max(population + jobs_wac - jobs_rac, population)`, whose LODES
  content is the net-commuter term `jobs_wac - jobs_rac`;
* the **retail-jobs leg**, `lodes_retail_jobs` (LODES CNS07), which enters larceny's opportunity
  hybrid at weight 0.2.

Both are frozen at their LODES vintage -- 2023 for 49 states, 2021 for Michigan, 2016 for Alaska
(outside the published universe). This module reads BLS QCEW annual-average county employment for
the vintage year and for the target year and forms a per-county ratio, which the exposure build
applies to the LODES legs before ensemble composition. LODES supplies the *within-county spatial
pattern*, which QCEW cannot: QCEW is a county-by-industry aggregate. QCEW supplies the *county
level* at the target year, which LODES cannot: the target year is not published yet.

Four rules are load-bearing:

* **This scales, it does not reallocate.** A single positive scalar per county multiplies that
  county's LODES legs. No block group's share of its county moves. The lane cannot invent or
  relocate an employer.
* **Fallback is a coverage ladder, not a guardrail stack.** County -> state -> identity, in that
  order, with the tier published per block group. The state rung exists because county vocabularies
  change under us: Connecticut's QCEW moved from eight counties (2023) to nine planning regions
  (2024), so no CT county has a county-to-county ratio across that boundary, while the CT *state*
  ratio is well defined in both.
* **The band is a disclosure guard, not a tuning knob.** Factors are clipped to
  [`SCALE_FACTOR_FLOOR`, `SCALE_FACTOR_CEILING`] = [0.5, 2.0] and every clipped county is logged.
  QCEW suppresses small county-industry cells (`disclosure_code == "N"`, published as 0), so a
  county whose retail cell is suppressed in one year and not the other produces a ratio that is a
  disclosure artifact rather than an employment change. Kauai HI 2023->2024 retail is 3,879 -> 8.
* **The level is mostly absorbed downstream.** Every named normalizer is rescaled to the reference
  universe's resident total, so a uniform national factor is a no-op. Only *cross-county
  dispersion* in the factor moves a published number.

No AGS value enters anywhere: the inputs are BLS QCEW and LEHD LODES.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.paths import RepoPaths


# --- naming and versioning ---------------------------------------------------------------

QCEW_SCALE_VERSION = "qcew_exposure_v1"
QCEW_TARGET_YEAR = 2024

# The semantics fragment PLAN item 6 requires the exposure lane to carry once its levels are no
# longer the LODES vintage's own.
MODELED_PLACE_EXPOSURE_PROXY = "modeled_place_exposure_proxy"

# QCEW industry codes. `10` is "Total, all industries"; `44-45` is the retail-trade NAICS sector,
# which is the QCEW counterpart of LODES CNS07 (the retail leg of larceny's hybrid).
QCEW_TOTAL_INDUSTRY_CODE = "10"
QCEW_RETAIL_INDUSTRY_CODE = "44-45"

# BLS's open-data URLs cannot carry the hyphen; the sector slug substitutes an underscore.
INDUSTRY_URL_SLUG = {
    QCEW_TOTAL_INDUSTRY_CODE: "10",
    QCEW_RETAIL_INDUSTRY_CODE: "44_45",
}

# QCEW aggregation levels (https://data.bls.gov/cew/doc/titles/agglevel/agglevel_titles.csv).
# 70 "County, Total Covered" is already summed over ownerships and carries one row per county;
# 74 "County, NAICS Sector -- by ownership sector" is split by ownership and must be summed.
# 50/54 are the state-level counterparts, which is what the coverage ladder's second rung reads.
AGGLVL_COUNTY_TOTAL = 70
AGGLVL_COUNTY_SECTOR = 74
AGGLVL_STATE_TOTAL = 50
AGGLVL_STATE_SECTOR = 54

INDUSTRY_AGGLVL = {
    QCEW_TOTAL_INDUSTRY_CODE: (AGGLVL_COUNTY_TOTAL, AGGLVL_STATE_TOTAL),
    QCEW_RETAIL_INDUSTRY_CODE: (AGGLVL_COUNTY_SECTOR, AGGLVL_STATE_SECTOR),
}

# The two scaled legs, named once so the frame columns, the summary and the contract agree.
TOTAL_LEG = "total"
RETAIL_LEG = "retail"
SCALED_LEGS: tuple[str, ...] = (TOTAL_LEG, RETAIL_LEG)
LEG_INDUSTRY_CODE = {
    TOTAL_LEG: QCEW_TOTAL_INDUSTRY_CODE,
    RETAIL_LEG: QCEW_RETAIL_INDUSTRY_CODE,
}

# The clip band. Wide enough that no genuine county employment change in CONUS reaches it over a
# one- or three-year update, narrow enough to catch a disclosure artifact.
SCALE_FACTOR_FLOOR = 0.5
SCALE_FACTOR_CEILING = 2.0

# The coverage ladder, most specific first. Published per block group so an artifact says which
# rung it stands on rather than presenting all three as the same measurement.
TIER_COUNTY = "county"
TIER_STATE = "state"
TIER_IDENTITY = "identity"
SCALE_TIERS: tuple[str, ...] = (TIER_COUNTY, TIER_STATE, TIER_IDENTITY)

# The LODES vintages present in the prepared block-group extract: 2023 for 49 states, 2021 for
# Michigan, 2016 for Alaska (which the release excludes anyway). Declared rather than discovered so
# the dependency list is stable, and CHECKED against the frame at build time -- a new vintage
# appearing is a QCEW file that needs fetching, and the build says so instead of silently
# publishing an unscaled county.
LODES_VINTAGE_YEARS: tuple[int, ...] = (2016, 2021, 2023)

QCEW_SOURCE = "BLS Quarterly Census of Employment and Wages, annual averages (open data access)"
QCEW_LICENSE = "public domain (U.S. Bureau of Labor Statistics)"

SCALE_FACTOR_COLUMNS = (
    "county_geoid",
    "source_year",
    "target_year",
    "leg",
    "source_employment",
    "target_employment",
    "raw_factor",
    "factor",
    "tier",
    "clipped",
)


def qcew_annual_url(*, year: int, industry_code: str) -> str:
    slug = INDUSTRY_URL_SLUG.get(str(industry_code))
    if slug is None:
        raise KeyError(f"Unknown QCEW industry code for this lane: {industry_code!r}")
    return f"https://data.bls.gov/cew/data/api/{int(year)}/a/industry/{slug}.csv"


def qcew_annual_path(paths: RepoPaths, *, year: int, industry_code: str) -> Path:
    slug = INDUSTRY_URL_SLUG.get(str(industry_code))
    if slug is None:
        raise KeyError(f"Unknown QCEW industry code for this lane: {industry_code!r}")
    return paths.data_dir / "qcew" / "raw" / f"qcew_{int(year)}_a_industry_{slug}.csv"


@dataclass(frozen=True)
class QcewScalingConfig:
    """Structural constants only. Nothing here is fitted, and nothing is chosen per result."""

    target_year: int = QCEW_TARGET_YEAR
    floor: float = SCALE_FACTOR_FLOOR
    ceiling: float = SCALE_FACTOR_CEILING

    def __post_init__(self) -> None:
        if not (0.0 < float(self.floor) <= 1.0 <= float(self.ceiling)):
            raise ValueError(
                f"the QCEW clip band must bracket 1.0 with a positive floor; got "
                f"[{self.floor}, {self.ceiling}]"
            )


# --- reading QCEW ------------------------------------------------------------------------


def _read_qcew_annual(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is a QCEW annual-average input for the exposure-updating lane and is absent. "
            f"Run `python scripts/pull/pull_qcew_county_employment.py` to fetch it from "
            f"{QCEW_SOURCE}."
        )
    frame = pd.read_csv(
        path,
        dtype={"area_fips": "string", "industry_code": "string", "disclosure_code": "string"},
        usecols=[
            "area_fips",
            "own_code",
            "industry_code",
            "agglvl_code",
            "year",
            "disclosure_code",
            "annual_avg_emplvl",
        ],
    )
    frame["area_fips"] = frame["area_fips"].astype("string").str.strip()
    frame["agglvl_code"] = pd.to_numeric(frame["agglvl_code"], errors="coerce")
    frame["annual_avg_emplvl"] = (
        pd.to_numeric(frame["annual_avg_emplvl"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    return frame


def _is_county_area(area_fips: pd.Series) -> pd.Series:
    """A real county-equivalent: five digits, and not a `999` statewide/unknown residual."""
    codes = area_fips.astype("string").fillna("")
    return codes.str.fullmatch(r"\d{5}").fillna(False) & ~codes.str.endswith("999").fillna(False)


def _is_state_area(area_fips: pd.Series) -> pd.Series:
    codes = area_fips.astype("string").fillna("")
    return codes.str.fullmatch(r"\d{2}000").fillna(False)


def qcew_employment(
    path: Path, *, industry_code: str, level: str
) -> pd.Series:
    """Annual-average employment by area, summed over ownership sectors.

    `level` is `"county"` or `"state"`. Ownership is summed because LODES JT00 counts every
    UI-covered job plus federal employment, so the comparable QCEW quantity is total covered
    employment rather than the private slice.
    """
    county_agglvl, state_agglvl = INDUSTRY_AGGLVL[str(industry_code)]
    if level == "county":
        agglvl, mask_fn = county_agglvl, _is_county_area
    elif level == "state":
        agglvl, mask_fn = state_agglvl, _is_state_area
    else:
        raise ValueError(f"level must be 'county' or 'state', got {level!r}")

    frame = _read_qcew_annual(path)
    frame = frame[frame["agglvl_code"].eq(float(agglvl))]
    frame = frame[mask_fn(frame["area_fips"])]
    if frame.empty:
        raise ValueError(
            f"{path} carries no {level}-level rows at aggregation level {agglvl} for industry "
            f"{industry_code!r}; the QCEW file is not the annual industry slice this lane reads."
        )
    key = frame["area_fips"].astype(str)
    if level == "state":
        key = key.str.slice(0, 2)
    return frame.assign(_key=key).groupby("_key")["annual_avg_emplvl"].sum().sort_index()


# --- factor construction -----------------------------------------------------------------


def scale_factor_from_totals(
    source: pd.Series, target: pd.Series, *, config: QcewScalingConfig = QcewScalingConfig()
) -> pd.Series:
    """target / source, defined only where both years carry positive employment.

    A zero on either side is a suppressed or non-existent cell, not an employment collapse, so it
    yields NaN and falls to the next rung of the coverage ladder rather than a 0 or an infinity.
    """
    joined = pd.concat({"source": source, "target": target}, axis=1)
    usable = joined["source"].gt(0.0) & joined["target"].gt(0.0)
    return (joined["target"] / joined["source"]).where(usable)


def clip_to_band(
    factors: pd.Series, *, config: QcewScalingConfig = QcewScalingConfig()
) -> tuple[pd.Series, pd.Series]:
    """Clip to the band and report which entries the band actually moved."""
    values = pd.to_numeric(factors, errors="coerce")
    clipped = values.clip(lower=float(config.floor), upper=float(config.ceiling))
    moved = values.notna() & ~np.isclose(values.to_numpy(dtype=float), clipped.to_numpy(dtype=float))
    return clipped, pd.Series(moved, index=values.index)


def build_county_scale_factors(
    *,
    paths: RepoPaths,
    source_year: int,
    config: QcewScalingConfig = QcewScalingConfig(),
) -> pd.DataFrame:
    """One row per (county, leg): the raw ratio, the banded factor, and which rung it came from.

    The state rung is applied to every county of a state whose own county ratio is undefined, which
    is what carries Connecticut across the 2024 planning-region renumbering. The identity rung is
    the last resort and means "this lane declined to move this county".
    """
    target_year = int(config.target_year)
    source_year = int(source_year)
    rows: list[pd.DataFrame] = []
    for leg in SCALED_LEGS:
        industry_code = LEG_INDUSTRY_CODE[leg]
        source_county = qcew_employment(
            qcew_annual_path(paths, year=source_year, industry_code=industry_code),
            industry_code=industry_code,
            level="county",
        )
        target_county = qcew_employment(
            qcew_annual_path(paths, year=target_year, industry_code=industry_code),
            industry_code=industry_code,
            level="county",
        )
        source_state = qcew_employment(
            qcew_annual_path(paths, year=source_year, industry_code=industry_code),
            industry_code=industry_code,
            level="state",
        )
        target_state = qcew_employment(
            qcew_annual_path(paths, year=target_year, industry_code=industry_code),
            industry_code=industry_code,
            level="state",
        )

        county_factor = scale_factor_from_totals(source_county, target_county, config=config)
        state_factor = scale_factor_from_totals(source_state, target_state, config=config)

        # The county universe is every county either year knows about, so a county that exists in
        # only one vintage still gets a published row (on the state or identity rung) instead of
        # silently vanishing.
        index = pd.Index(
            sorted(set(source_county.index) | set(target_county.index)), name="county_geoid"
        )
        frame = pd.DataFrame(index=index)
        frame.index.name = "county_geoid"
        frame["source_employment"] = source_county.reindex(index).fillna(0.0)
        frame["target_employment"] = target_county.reindex(index).fillna(0.0)
        state_of_county = pd.Series(index.str.slice(0, 2), index=index)
        state_ratio = state_of_county.map(state_factor)

        raw = county_factor.reindex(index)
        tier = pd.Series(TIER_COUNTY, index=index, dtype="object")
        tier = tier.where(raw.notna(), TIER_STATE)
        raw = raw.where(raw.notna(), state_ratio)
        tier = tier.where(raw.notna(), TIER_IDENTITY)
        raw = raw.fillna(1.0)

        factor, clipped = clip_to_band(raw, config=config)
        frame["source_year"] = source_year
        frame["target_year"] = target_year
        frame["leg"] = leg
        frame["raw_factor"] = raw.astype(float)
        frame["factor"] = factor.astype(float)
        frame["tier"] = tier.astype(str)
        frame["clipped"] = clipped.fillna(False).astype(bool)
        rows.append(frame.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out["county_geoid"] = out["county_geoid"].astype("string").str.zfill(5)
    out = out.reindex(columns=list(SCALE_FACTOR_COLUMNS))
    return out.sort_values(["leg", "county_geoid"], kind="mergesort").reset_index(drop=True)


def build_scale_factor_table(
    *,
    paths: RepoPaths,
    source_years: tuple[int, ...],
    config: QcewScalingConfig = QcewScalingConfig(),
) -> pd.DataFrame:
    """The factor table over every LODES vintage present in the block-group universe."""
    if not source_years:
        raise ValueError("the QCEW exposure lane needs at least one LODES vintage year")
    frames = [
        build_county_scale_factors(paths=paths, source_year=int(year), config=config)
        for year in sorted({int(year) for year in source_years})
    ]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["source_year", "leg", "county_geoid"], kind="mergesort").reset_index(
        drop=True
    )


def assert_scale_factor_invariants(
    factors: pd.DataFrame, *, config: QcewScalingConfig = QcewScalingConfig()
) -> None:
    missing = [column for column in SCALE_FACTOR_COLUMNS if column not in factors.columns]
    if missing:
        raise ValueError(f"QCEW scale-factor table is missing columns {missing}")
    if factors.duplicated(["source_year", "leg", "county_geoid"]).any():
        raise ValueError("QCEW scale-factor table carries duplicate (source_year, leg, county) rows")
    values = pd.to_numeric(factors["factor"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("QCEW scale-factor table carries a nonfinite factor")
    if values.lt(float(config.floor) - 1e-12).any() or values.gt(float(config.ceiling) + 1e-12).any():
        raise ValueError(
            f"QCEW scale-factor table carries a factor outside the band "
            f"[{config.floor}, {config.ceiling}]"
        )
    unknown_tier = set(factors["tier"].astype(str)) - set(SCALE_TIERS)
    if unknown_tier:
        raise ValueError(f"QCEW scale-factor table carries unknown tiers {sorted(unknown_tier)}")
    identity = factors["tier"].astype(str).eq(TIER_IDENTITY)
    if identity.any() and not np.allclose(values[identity].to_numpy(dtype=float), 1.0):
        raise ValueError("QCEW scale-factor table has an identity-tier row whose factor is not 1.0")
    unknown_leg = set(factors["leg"].astype(str)) - set(SCALED_LEGS)
    if unknown_leg:
        raise ValueError(f"QCEW scale-factor table carries unknown legs {sorted(unknown_leg)}")


# --- application -------------------------------------------------------------------------


def county_geoid_from_bg(bg_id: pd.Series) -> pd.Series:
    return bg_id.astype("string").str.zfill(12).str.slice(0, 5)


def attach_scale_factors(
    frame: pd.DataFrame,
    *,
    factors: pd.DataFrame,
    lodes_vintage: pd.Series,
    config: QcewScalingConfig = QcewScalingConfig(),
) -> pd.DataFrame:
    """Per block group, the factor and tier of each scaled leg, keyed on (county, LODES vintage).

    A block group whose county-vintage pair has no row takes the identity factor rather than a NaN:
    an absent factor means "no measured level change", never "no exposure".
    """
    assert_scale_factor_invariants(factors, config=config)
    county = county_geoid_from_bg(frame["bg_id"])
    vintage = pd.to_numeric(lodes_vintage, errors="coerce").astype("Int64")
    out = pd.DataFrame(index=frame.index)
    out["county_geoid"] = county
    out["lodes_vintage"] = vintage
    out["qcew_scale_vintage"] = int(config.target_year)
    key = pd.DataFrame({"county_geoid": county.astype(str), "source_year": vintage.astype("float")})
    for leg in SCALED_LEGS:
        leg_rows = factors[factors["leg"].astype(str).eq(leg)][
            ["county_geoid", "source_year", "factor", "tier"]
        ].copy()
        leg_rows["county_geoid"] = leg_rows["county_geoid"].astype("string").str.zfill(5).astype(str)
        leg_rows["source_year"] = pd.to_numeric(leg_rows["source_year"], errors="coerce").astype(float)
        merged = key.merge(leg_rows, on=["county_geoid", "source_year"], how="left", validate="many_to_one")
        out[f"qcew_{leg}_scale_factor"] = (
            pd.to_numeric(merged["factor"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        )
        out[f"qcew_{leg}_scale_tier"] = (
            merged["tier"].astype("string").fillna(TIER_IDENTITY).astype(str).to_numpy()
        )
    return out


def scale_daytime_jobs_leg(
    *,
    population: np.ndarray,
    jobs_wac: np.ndarray,
    jobs_rac: np.ndarray,
    factor: np.ndarray,
) -> np.ndarray:
    """`max(population + f * (jobs_wac - jobs_rac), population)`, the leg's own formula updated.

    Only the LODES-derived term moves. The resident term is an ACS/decennial quantity already at
    the release vintage, and multiplying it by an employment ratio would be a category error --
    which is exactly why this scales the net-commuter term rather than the assembled leg. The floor
    at resident population is the deployed leg's own floor, kept so the scaled leg stays a
    person-scale surface that a convex mix with LandScan can be taken of.
    """
    people = np.clip(np.nan_to_num(np.asarray(population, dtype=float), nan=0.0), 0.0, None)
    wac = np.clip(np.nan_to_num(np.asarray(jobs_wac, dtype=float), nan=0.0), 0.0, None)
    rac = np.clip(np.nan_to_num(np.asarray(jobs_rac, dtype=float), nan=0.0), 0.0, None)
    scale = np.nan_to_num(np.asarray(factor, dtype=float), nan=1.0)
    return np.maximum(np.clip(people + scale * (wac - rac), 0.0, None), people)


def scale_retail_leg(*, retail_jobs: np.ndarray, factor: np.ndarray) -> np.ndarray:
    values = np.clip(np.nan_to_num(np.asarray(retail_jobs, dtype=float), nan=0.0), 0.0, None)
    scale = np.nan_to_num(np.asarray(factor, dtype=float), nan=1.0)
    return np.clip(values * scale, 0.0, None)


# --- reporting ---------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "p5": float("nan"), "p95": float("nan")}
    return {
        "median": float(np.median(values)),
        "p5": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_scale_factors(
    factors: pd.DataFrame, *, config: QcewScalingConfig = QcewScalingConfig()
) -> dict:
    """The distribution, per (vintage, leg). Reported so "this barely matters" stays falsifiable."""
    summary: dict[str, object] = {
        "scale_version": QCEW_SCALE_VERSION,
        "target_year": int(config.target_year),
        "band": [float(config.floor), float(config.ceiling)],
        "source": QCEW_SOURCE,
        "license": QCEW_LICENSE,
        "ags_values_used": False,
        "by_source_year": {},
    }
    for (source_year, leg), group in factors.groupby(["source_year", "leg"], sort=True):
        raw = pd.to_numeric(group["raw_factor"], errors="coerce").to_numpy(dtype=float)
        banded = pd.to_numeric(group["factor"], errors="coerce").to_numpy(dtype=float)
        tiers = group["tier"].astype(str)
        entry = {
            "counties": int(len(group)),
            "raw_factor": _quantiles(raw),
            "factor": _quantiles(banded),
            "min_raw_factor": float(raw.min()) if raw.size else float("nan"),
            "max_raw_factor": float(raw.max()) if raw.size else float("nan"),
            "out_of_band_counties": int(group["clipped"].astype(bool).sum()),
            "out_of_band_county_geoids": sorted(
                group.loc[group["clipped"].astype(bool), "county_geoid"].astype(str).tolist()
            ),
            "within_1pct_of_unity": float(np.mean(np.abs(banded - 1.0) <= 0.01)) if banded.size else float("nan"),
            "within_5pct_of_unity": float(np.mean(np.abs(banded - 1.0) <= 0.05)) if banded.size else float("nan"),
            "tier_counts": {tier: int((tiers == tier).sum()) for tier in SCALE_TIERS},
        }
        bucket = summary["by_source_year"].setdefault(str(int(source_year)), {})  # type: ignore[union-attr]
        bucket[str(leg)] = entry
    return summary


def summarize_applied_factors(applied: pd.DataFrame) -> dict:
    """What the build actually applied, block group by block group. A record, not a re-derivation."""
    summary: dict[str, object] = {
        "block_groups": int(len(applied)),
        "lodes_vintages": {
            str(int(year)): int(count)
            for year, count in applied["lodes_vintage"].dropna().astype(int).value_counts().sort_index().items()
        },
        "qcew_scale_vintage": int(pd.to_numeric(applied["qcew_scale_vintage"]).max())
        if len(applied)
        else None,
        "legs": {},
    }
    for leg in SCALED_LEGS:
        values = pd.to_numeric(applied[f"qcew_{leg}_scale_factor"], errors="coerce").to_numpy(dtype=float)
        tiers = applied[f"qcew_{leg}_scale_tier"].astype(str)
        summary["legs"][leg] = {  # type: ignore[index]
            **_quantiles(values),
            "min": float(values.min()) if values.size else float("nan"),
            "max": float(values.max()) if values.size else float("nan"),
            "tier_block_groups": {tier: int((tiers == tier).sum()) for tier in SCALE_TIERS},
        }
    return summary
