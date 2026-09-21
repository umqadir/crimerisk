"""v2 count-first composites: aggregates built from EXPECTED COUNTS under ONE COMMON DENOMINATOR.

`REVIEW_SOL_NEUTRAL.md` sec.4 rules out the construction the surface has been publishing: an
average of per-offense indexes whose denominators differ. Burglary per premises, assault per
person presence and motor vehicle theft per vehicle have no common rate unit, so their weighted
average is a dimensionless dashboard score and not a rate of anything. The v2 lane replaces it
with two composites that are rates:

```
event burden   I_g = 100 * ( (Sum_o C_go)       / P_g ) / ( (Sum_o C_USo)       / P_US )
harm  burden   I_g = 100 * ( (Sum_o h_o C_go)   / P_g ) / ( (Sum_o h_o C_USo)   / P_US )
```

with `P` = resident population, one denominator for every offense, and `h_o` a versioned,
publicly documented severity vector read from `configs/severity_weights_v1.csv`.

Four rules are load-bearing and enforced rather than trusted:

* **Count-first means denominator-blind ACROSS FAMILIES, not suppression-blind.** These composites
  never read a per-offense opportunity denominator, so a cell whose *vehicle* exposure is below
  its floor still publishes a resident burden -- that is the substantive difference from the
  fields they replace. They are still all-or-null over their own components on their own
  denominator family: a component whose RESIDENT rate the surface suppresses is a component this
  composite may not assert, so the composite goes null with it.
* **The harm composite publishes at TRACT support and coarser, never at block group.** Roughly
  half the national harm mass sits on murder and rape under any sentencing-style vector, and this
  project already refuses to publish a block-group murder point because a single year of it is
  Poisson noise on a model prior. A block-group harm index would republish exactly that noise,
  amplified 5,475x. The event burden has no such problem: it weights every event equally, so a
  rare offense enters at its (tiny) count share.
* **Severity is a normative choice, not an estimated parameter.** The vector, its source, its
  version and a sensitivity result under alternative public vectors ship with the build. The
  index is invariant to any positive rescaling of the whole vector, so a days vector and a
  dollars vector are directly comparable.
* **A mixed-denominator average may be retained for research but not called a total.** The
  consult's wording: name it `multi_offense_relative_score`, not `total_rate`. This lane renames
  the two surviving index averages and changes not one of their values.

No AGS value enters anywhere: the composites are functions of this project's own counts, the
Census resident population, and a published severity vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimerisk.crime import OFFENSES_7
from crimerisk.paths import RepoPaths
from crimerisk.special_use import SPECIAL_USE_RESIDENT_ALLOWED_COLUMN


# --- naming and versioning ---------------------------------------------------------------

COMPOSITE_VERSION = "count_first_composites_v1"

# The one denominator every count-first composite divides by. Named, because "which denominator"
# is the whole question sec.4 was answering.
COMMON_DENOMINATOR_ID = "resident_population_v1"
COMMON_DENOMINATOR_COLUMN = "resident_secondary_denominator"
COMMON_DENOMINATOR_SEMANTICS = "recorded_offense_burden_per_resident_not_personal_danger"

EVENT_BURDEN_COLUMN = "index_event_burden_resident"
PERSONAL_BURDEN_COLUMN = "index_personal_burden_resident"
PROPERTY_BURDEN_COLUMN = "index_property_burden_resident"
HARM_BURDEN_COLUMN = "index_harm_burden_resident"
HARM_WEIGHTED_COUNT_COLUMN = "harm_weighted_count_total"
PERSONAL_RELATIVE_SCORE_COLUMN = "multi_offense_relative_score_personal_event_weighted"
PROPERTY_RELATIVE_SCORE_COLUMN = "multi_offense_relative_score_property_event_weighted"

# Support policy. The harm index is published at tract and coarser only; the block-group frame
# carries the column and leaves it null, the same shape the rare-offense publication policy uses
# for murder/rape.
BLOCK_GROUP_SUPPORT = "block_group"
TRACT_SUPPORT = "tract"
HARM_BURDEN_MIN_SUPPORT = TRACT_SUPPORT

# The offense groups the three event-scale burdens run over. These mirror
# `allocation.PERSONAL_OFFENSES` / `allocation.PROPERTY_OFFENSES`; the pairing is asserted in
# tests rather than left to drift, because allocation cannot be imported here without a cycle.
PERSONAL_OFFENSES: tuple[str, ...] = ("murder", "rape", "robbery", "aggravated_assault")
PROPERTY_OFFENSES: tuple[str, ...] = ("burglary", "larceny", "motor_vehicle_theft")
BURDEN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (EVENT_BURDEN_COLUMN, tuple(OFFENSES_7)),
    (PERSONAL_BURDEN_COLUMN, PERSONAL_OFFENSES),
    (PROPERTY_BURDEN_COLUMN, PROPERTY_OFFENSES),
)

# Legacy field -> v2 field. The first four are superseded constructions; the last two are the
# consult's rename, values untouched.
SUPERSEDED_COMPOSITE_FIELDS: dict[str, str] = {
    "index_total_part1_resident": EVENT_BURDEN_COLUMN,
    "index_personal_part1_resident": PERSONAL_BURDEN_COLUMN,
    "index_property_part1_resident": PROPERTY_BURDEN_COLUMN,
    "index_total_harm": HARM_BURDEN_COLUMN,
}
RENAMED_COMPOSITE_FIELDS: dict[str, str] = {
    "index_total_primary_event_weighted": "multi_offense_relative_score_event_weighted",
    "index_total_equal_offense": "multi_offense_relative_score_equal_offense",
}

LEGACY_AGGREGATE_INDEX_FIELDS: tuple[str, ...] = (
    "index_total_part1_resident",
    "index_personal_part1_resident",
    "index_property_part1_resident",
    "index_total_primary_event_weighted",
    "index_total_equal_offense",
    "index_total_harm",
)
COUNT_FIRST_AGGREGATE_INDEX_FIELDS: tuple[str, ...] = (
    EVENT_BURDEN_COLUMN,
    PERSONAL_BURDEN_COLUMN,
    PROPERTY_BURDEN_COLUMN,
    HARM_BURDEN_COLUMN,
    *RENAMED_COMPOSITE_FIELDS.values(),
    PERSONAL_RELATIVE_SCORE_COLUMN,
    PROPERTY_RELATIVE_SCORE_COLUMN,
)

RATE_PER_100K = 100000.0

# Publication rule for every count-first composite. Deliberately identical across them and
# deliberately free of any per-offense denominator term: these are the conditions under which the
# COMMON denominator is a usable measure of who is there. Mirrors the constants in allocation.py
# (`NON_RESIDENTIAL_HOUSEHOLD_FLOOR`, `PERSON_EXPOSURE_DENOMINATOR_FLOOR`); asserted equal in
# tests rather than imported, to keep this module free of an allocation import cycle.
NON_RESIDENTIAL_HOUSEHOLD_FLOOR = 10.0
COMMON_DENOMINATOR_FLOOR = 50.0

# The published map break set -- log-symmetric about 100. "Changing two or more map bins" in the
# severity sensitivity is measured against exactly these edges. The retired frontend carried the
# same list as a literal `INDEX_BREAKS` and a test parsed it back; the current map reads the break
# set from the edition manifest instead, where the release validator checks it.
INDEX_MAP_BREAKS: tuple[float, ...] = (
    3.125,
    6.25,
    12.5,
    25.0,
    50.0,
    75.0,
    100.0,
    133.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12800.0,
)

SEVERITY_WEIGHTS_FILENAME = "severity_weights_v1.csv"
SEVERITY_TABLE_COLUMNS = (
    "vector_id",
    "role",
    "offense",
    "weight",
    "unit",
    "source",
    "source_version",
    "note",
)
SEVERITY_ROLES = ("primary", "alternative", "anchor")
# A derived vector, not a published one: the primary vector with the two offenses whose
# block-group points this project refuses to publish set to zero. It answers the consult's
# "extent to which rare-offense allocation determines the result" with a number.
VOLUME_ONLY_SUFFIX = "_volume_only"
RARE_OFFENSES: tuple[str, ...] = ("murder", "rape")


def aggregate_index_fields(*, count_first: bool) -> tuple[str, ...]:
    return COUNT_FIRST_AGGREGATE_INDEX_FIELDS if count_first else LEGACY_AGGREGATE_INDEX_FIELDS


def multi_offense_score_column(legacy_field: str, *, count_first: bool) -> str:
    """The column an index-average composite is published under on each lane."""
    if legacy_field not in RENAMED_COMPOSITE_FIELDS:
        raise KeyError(f"{legacy_field!r} is not one of the index-average composites")
    return RENAMED_COMPOSITE_FIELDS[legacy_field] if count_first else legacy_field


def harm_index_is_supported(support: str) -> bool:
    """True at tract and coarser, false at block group. See the module docstring."""
    return str(support) != BLOCK_GROUP_SUPPORT


def support_for_geo_id_col(geo_id_col: str) -> str:
    return BLOCK_GROUP_SUPPORT if str(geo_id_col) == "block_group_geoid" else TRACT_SUPPORT


# --- paths -------------------------------------------------------------------------------


def severity_weights_path(paths: RepoPaths) -> Path:
    return paths.repo_root / "configs" / SEVERITY_WEIGHTS_FILENAME


def severity_sensitivity_path(directory: Path, *, year: int) -> Path:
    return directory / f"composite_severity_sensitivity_{int(year)}.csv"


# --- the severity vector table -----------------------------------------------------------


def load_severity_weights(path: Path) -> pd.DataFrame:
    """Read the versioned severity vectors and refuse anything that is not a usable vector.

    This is a read, not a selection. Severity weights are a normative choice made once and
    published; a corrupted or hand-edited table must fail the build rather than silently reweight
    every harm number in the country.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is the severity-vector source of record and is absent. It carries the "
            "primary vector, its citation, and the alternative public vectors the harm "
            "composite's sensitivity result is computed against."
        )
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = [column for column in SEVERITY_TABLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"severity weight table {path} is missing columns {missing}")
    for column in ("vector_id", "role", "offense", "unit", "source", "source_version"):
        frame[column] = frame[column].astype("string").fillna("").str.strip()
    frame["note"] = frame["note"].astype("string").fillna("")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").astype(float)

    if frame.duplicated(["vector_id", "offense"]).any():
        raise ValueError(f"severity weight table {path} carries duplicate (vector_id, offense) rows")
    if frame["weight"].isna().any():
        raise ValueError(f"severity weight table {path} carries a non-numeric weight")
    if not np.isfinite(frame["weight"].to_numpy(dtype=float)).all():
        raise ValueError(f"severity weight table {path} carries a non-finite weight")
    if frame["weight"].le(0.0).any():
        raise ValueError(
            f"severity weight table {path} carries a non-positive weight; a zero weight silently "
            "drops an offense out of the harm composite and must be a separate, named vector"
        )
    bad_roles = sorted(set(frame["role"].tolist()) - set(SEVERITY_ROLES))
    if bad_roles:
        raise ValueError(f"severity weight table {path} carries unknown roles {bad_roles}")

    for vector_id, rows in frame.groupby("vector_id", sort=True):
        offenses = set(rows["offense"].tolist())
        if offenses != set(OFFENSES_7):
            missing_offenses = sorted(set(OFFENSES_7) - offenses)
            unexpected = sorted(offenses - set(OFFENSES_7))
            raise ValueError(
                f"severity vector {vector_id!r} in {path} must cover exactly the seven Part-I "
                f"offenses: missing {missing_offenses}, unexpected {unexpected}"
            )
        for column in ("role", "unit", "source", "source_version"):
            values = set(rows[column].tolist())
            if len(values) != 1:
                raise ValueError(
                    f"severity vector {vector_id!r} in {path} carries {len(values)} distinct "
                    f"{column} values {sorted(values)}; a vector is one choice with one citation"
                )
        if str(rows["source"].iloc[0]) == "":
            raise ValueError(f"severity vector {vector_id!r} in {path} carries no source")

    primaries = sorted(set(frame.loc[frame["role"].eq("primary"), "vector_id"].tolist()))
    if len(primaries) != 1:
        raise ValueError(
            f"severity weight table {path} must name exactly one primary vector, found {primaries}"
        )
    alternatives = sorted(set(frame.loc[frame["role"].eq("alternative"), "vector_id"].tolist()))
    if not alternatives:
        raise ValueError(
            f"severity weight table {path} carries no alternative vector; the harm composite "
            "does not publish without a sensitivity result under at least one alternative "
            "(REVIEW_SOL_NEUTRAL.md sec.4)"
        )
    return frame.sort_values(["vector_id", "offense"], kind="mergesort").reset_index(drop=True)


def primary_vector_id(weights: pd.DataFrame) -> str:
    return str(weights.loc[weights["role"].eq("primary"), "vector_id"].iloc[0])


def vector_ids(weights: pd.DataFrame, *, role: str | None = None) -> tuple[str, ...]:
    rows = weights if role is None else weights[weights["role"].eq(role)]
    return tuple(sorted(set(str(value) for value in rows["vector_id"].tolist())))


def severity_vector(weights: pd.DataFrame, vector_id: str) -> dict[str, float]:
    rows = weights[weights["vector_id"].eq(str(vector_id))]
    if rows.empty:
        raise KeyError(f"severity vector {vector_id!r} is not in the weight table")
    return {str(row.offense): float(row.weight) for row in rows.itertuples()}


def vector_metadata(weights: pd.DataFrame, vector_id: str) -> dict[str, object]:
    rows = weights[weights["vector_id"].eq(str(vector_id))]
    if rows.empty:
        raise KeyError(f"severity vector {vector_id!r} is not in the weight table")
    first = rows.iloc[0]
    return {
        "vector_id": str(vector_id),
        "role": str(first["role"]),
        "unit": str(first["unit"]),
        "source": str(first["source"]),
        "source_version": str(first["source_version"]),
        "weights": severity_vector(weights, vector_id),
    }


def volume_only_vector(vector: dict[str, float]) -> dict[str, float]:
    """The same vector with the rare offenses zeroed -- a diagnostic, never a published vector."""
    return {offense: (0.0 if offense in RARE_OFFENSES else float(weight)) for offense, weight in vector.items()}


# --- composite arithmetic ----------------------------------------------------------------


def _nonnegative(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def expected_count_column(offense: str) -> str:
    return f"expected_count_{str(offense)}"


def weighted_count(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Sum_o w_o * expected_count_o. One pass over published counts; nothing else is read."""
    total = pd.Series(0.0, index=frame.index, dtype=float)
    for offense, weight in weights.items():
        if float(weight) == 0.0:
            continue
        total = total + float(weight) * _nonnegative(frame[expected_count_column(offense)])
    return total


def gated_component_offenses(offenses: tuple[str, ...], support: str) -> tuple[str, ...]:
    """The components a burden at this support may be vetoed by.

    Murder and rape carry no per-cell index at block-group support: the rare-offense policy
    withholds the POINT there because one year of murder on 1,500 people is Poisson noise. That
    is a support rule, not a suppression -- the counts are real, conserved, and are exactly what
    these count-first composites consume. There is therefore nothing to gate on, and reading the
    withheld flag would null every block group in the country. At tract support the rare offenses
    publish normally and are gated like any other component.
    """
    if str(support) != BLOCK_GROUP_SUPPORT:
        return tuple(offenses)
    return tuple(offense for offense in offenses if offense not in RARE_OFFENSES)


def resident_component_publishable(frame: pd.DataFrame, offenses: tuple[str, ...]) -> pd.Series:
    """True where every component offense is publishable on the RESIDENT denominator family.

    A composite must not assert a component the surface itself suppresses. The family matters:
    `estimate_mode_*` and `index_*_primary` belong to the per-offense OPPORTUNITY denominators
    (premises, vehicles, person-exposure) and say nothing about resident population, so they are
    deliberately not read here -- that is still the mixed-denominator confusion this lane exists
    to remove. What is read is each component's own resident-arm gate, which is a statement about
    the very denominator these composites divide by.

    A surface built before the resident gates existed carries none of these columns; it then
    reduces to the denominator test alone, exactly as before.
    """
    publishable = pd.Series(True, index=frame.index)
    for offense in offenses:
        column = f"index_{offense}_resident_publishable"
        if column not in frame.columns:
            continue
        publishable &= pd.Series(frame[column], index=frame.index).fillna(False).astype(bool)
    return publishable


def burden_publishable(
    frame: pd.DataFrame, offenses: tuple[str, ...] | None = None
) -> pd.Series:
    """Where the COMMON denominator is a usable measure of who is there.

    No per-offense denominator appears here on purpose: a count-first composite that inherited
    the vehicle-exposure floor would be exactly the mixed-denominator confusion this lane exists
    to remove. `offenses` adds the one component test that IS on this composite's own
    denominator family (see `resident_component_publishable`); omitting it keeps the plain
    denominator rule, which is what the normalizer record and the sensitivity matrix want.

    The special-use term follows whichever rule the surface was built under. The composites
    divide by resident population, so the term they take from the typed lane is its RESIDENT gate
    (`special_use_resident_rate_allowed`) and not its exposure gate -- an employment district
    publishes an exposure rate and no resident burden, which is the same statement in both places.
    On an ordinary cell the typed gate reduces to `households >= 10` exactly, so the two
    expressions agree everywhere the taxonomy was not asked.
    """
    denominator = _nonnegative(frame[COMMON_DENOMINATOR_COLUMN])
    components = (
        resident_component_publishable(frame, offenses)
        if offenses
        else pd.Series(True, index=frame.index)
    )
    if SPECIAL_USE_RESIDENT_ALLOWED_COLUMN in frame.columns:
        residential = (
            pd.Series(frame[SPECIAL_USE_RESIDENT_ALLOWED_COLUMN], index=frame.index)
            .fillna(False)
            .astype(bool)
        )
        return (
            residential
            & denominator.gt(0.0)
            & denominator.ge(float(COMMON_DENOMINATOR_FLOOR))
            & components
        )
    households = _nonnegative(frame["households_total"])
    special_use = (
        pd.Series(frame["special_use_tract_flag"], index=frame.index).fillna(False).astype(bool)
        if "special_use_tract_flag" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    return (
        households.ge(float(NON_RESIDENTIAL_HOUSEHOLD_FLOOR))
        & denominator.gt(0.0)
        & denominator.ge(float(COMMON_DENOMINATOR_FLOOR))
        & ~special_use
        & components
    )


def count_first_index(
    *,
    counts: pd.Series,
    denominator: pd.Series,
    publishable: pd.Series,
) -> dict[str, object]:
    """100 * (count/denominator) / (Sum count / Sum denominator) over the publishable rows."""
    denom = _nonnegative(denominator)
    count = _nonnegative(counts)
    pub = pd.Series(publishable, index=count.index).fillna(False).astype(bool) & denom.gt(0.0)
    denom_sum = float(denom.loc[pub].sum())
    count_sum = float(count.loc[pub].sum())
    reference_rate = RATE_PER_100K * count_sum / denom_sum if denom_sum > 0.0 else float("nan")
    rate = pd.Series(np.nan, index=count.index, dtype=float)
    rate.loc[pub] = RATE_PER_100K * count.loc[pub] / denom.loc[pub]
    index = pd.Series(np.nan, index=count.index, dtype=float)
    if np.isfinite(reference_rate) and reference_rate > 0.0:
        index.loc[pub] = 100.0 * rate.loc[pub] / reference_rate
    return {
        "rate": rate.replace([np.inf, -np.inf], np.nan),
        "index": index.replace([np.inf, -np.inf], np.nan),
        "reference_rate_per_100k": reference_rate,
        "publishable": pub,
        "count_total": count_sum,
        "denominator_total": denom_sum,
    }


def harm_burden_index(
    frame: pd.DataFrame,
    *,
    vector: dict[str, float],
    publishable: pd.Series | None = None,
) -> dict[str, object]:
    pub = burden_publishable(frame) if publishable is None else publishable
    return count_first_index(
        counts=weighted_count(frame, vector),
        denominator=frame[COMMON_DENOMINATOR_COLUMN],
        publishable=pub,
    )


# --- map bins and the severity sensitivity matrix ------------------------------------------


def map_bin(values: pd.Series, breaks: tuple[float, ...] = INDEX_MAP_BREAKS) -> pd.Series:
    """The published map's bin ordinal for an index value: 0 below the first break, len(breaks)
    at or above the last. The map interpolates colour within a bin; the bin is the unit a reader
    can actually distinguish, so it is the unit the sensitivity result is quoted in."""
    numeric = pd.to_numeric(values, errors="coerce")
    ordinal = pd.Series(
        np.searchsorted(np.asarray(breaks, dtype=float), numeric.fillna(-1.0).to_numpy(dtype=float), side="right"),
        index=numeric.index,
        dtype="float64",
    )
    return ordinal.where(numeric.notna())


def compare_severity_vectors(
    primary_index: pd.Series,
    alternative_index: pd.Series,
) -> dict[str, object]:
    """Rank correlation, top-decile overlap and map-bin movement between two harm surfaces."""
    both = pd.to_numeric(primary_index, errors="coerce").notna() & pd.to_numeric(
        alternative_index, errors="coerce"
    ).notna()
    left = pd.to_numeric(primary_index, errors="coerce").loc[both]
    right = pd.to_numeric(alternative_index, errors="coerce").loc[both]
    rows = int(both.sum())
    result: dict[str, object] = {
        "rows_compared": rows,
        "spearman_rho": float("nan"),
        "top_decile_overlap": float("nan"),
        "share_moving_1plus_bins": float("nan"),
        "share_moving_2plus_bins": float("nan"),
        "max_abs_bin_shift": float("nan"),
    }
    if rows == 0:
        return result
    if rows > 1 and left.nunique() > 1 and right.nunique() > 1:
        result["spearman_rho"] = float(left.corr(right, method="spearman"))
    cutoff = max(1, int(round(0.1 * rows)))
    top_left = set(left.rank(method="first", ascending=False).nsmallest(cutoff).index)
    top_right = set(right.rank(method="first", ascending=False).nsmallest(cutoff).index)
    result["top_decile_overlap"] = float(len(top_left & top_right) / cutoff)
    shift = (map_bin(right) - map_bin(left)).abs()
    result["share_moving_1plus_bins"] = float((shift >= 1).mean())
    result["share_moving_2plus_bins"] = float((shift >= 2).mean())
    result["max_abs_bin_shift"] = float(shift.max())
    return result


def severity_sensitivity(
    frame: pd.DataFrame,
    *,
    weights: pd.DataFrame,
    label: str,
    support: str,
) -> pd.DataFrame:
    """One row per comparison vector: how much of the harm ranking is the severity choice.

    Computed at the support the harm composite actually publishes at. Running it at block group
    would measure the sensitivity of a surface this lane refuses to publish.
    """
    publishable = burden_publishable(
        frame, gated_component_offenses(tuple(OFFENSES_7), support)
    )
    primary_id = primary_vector_id(weights)
    primary = severity_vector(weights, primary_id)
    primary_index = pd.Series(
        harm_burden_index(frame, vector=primary, publishable=publishable)["index"], index=frame.index
    )
    primary_mass = _vector_mass_shares(frame, primary)

    comparisons: list[tuple[str, str, dict[str, float]]] = [
        (vector_id, str(vector_metadata(weights, vector_id)["role"]), severity_vector(weights, vector_id))
        for vector_id in vector_ids(weights)
        if vector_id != primary_id
    ]
    comparisons.append((f"{primary_id}{VOLUME_ONLY_SUFFIX}", "diagnostic", volume_only_vector(primary)))

    rows: list[dict[str, object]] = []
    for vector_id, role, vector in comparisons:
        alternative_index = pd.Series(
            harm_burden_index(frame, vector=vector, publishable=publishable)["index"], index=frame.index
        )
        comparison = compare_severity_vectors(primary_index, alternative_index)
        rows.append(
            {
                "surface": label,
                "support": str(support),
                "primary_vector_id": primary_id,
                "comparison_vector_id": vector_id,
                "comparison_role": role,
                "published_rows": int(publishable.sum()),
                **comparison,
                "primary_rare_offense_harm_mass_share": float(
                    sum(primary_mass.get(offense, 0.0) for offense in RARE_OFFENSES)
                ),
                "primary_larceny_harm_mass_share": float(primary_mass.get("larceny", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _vector_mass_shares(frame: pd.DataFrame, vector: dict[str, float]) -> dict[str, float]:
    """Each offense's share of the surface's total weighted mass under `vector`."""
    masses = {
        offense: float(weight) * float(_nonnegative(frame[expected_count_column(offense)]).sum())
        for offense, weight in vector.items()
    }
    total = float(sum(masses.values()))
    if not np.isfinite(total) or total <= 0.0:
        return {offense: float("nan") for offense in masses}
    return {offense: value / total for offense, value in masses.items()}


# --- runtime -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeRuntime:
    """Everything the build needs to publish the count-first composites, resolved once."""

    weights: pd.DataFrame
    weights_path: Path
    year: int = 2024

    @property
    def primary_vector_id(self) -> str:
        return primary_vector_id(self.weights)

    @property
    def primary_vector(self) -> dict[str, float]:
        return severity_vector(self.weights, self.primary_vector_id)

    @property
    def alternative_vector_ids(self) -> tuple[str, ...]:
        return tuple(
            vector_id for vector_id in vector_ids(self.weights) if vector_id != self.primary_vector_id
        )


def resolve_composite_runtime(
    *,
    paths: RepoPaths,
    year: int = 2024,
    weights_path: Path | None = None,
) -> CompositeRuntime:
    path = weights_path or severity_weights_path(paths)
    return CompositeRuntime(weights=load_severity_weights(path), weights_path=path, year=int(year))


def apply_count_first_composites(
    frame: pd.DataFrame,
    *,
    runtime: CompositeRuntime,
    support: str,
) -> pd.DataFrame:
    """Publish the count-first burdens on a finalized surface.

    Called after the transient-exposure fold, from published fields only, so the values a
    consumer recomputes from the frame are the values the frame carries. The fold cannot move
    them -- it changes per-offense points and publication modes, and these read neither -- which
    is asserted rather than assumed in the tests.
    """
    out = frame.copy()
    # One publication rule per composite, evaluated over that composite's OWN offense set: a
    # property burden is not withheld because a personal offense was suppressed, and neither is
    # published while one of its own components is.
    for field, offenses in BURDEN_SPECS:
        published = count_first_index(
            counts=weighted_count(out, {offense: 1.0 for offense in offenses}),
            denominator=out[COMMON_DENOMINATOR_COLUMN],
            publishable=burden_publishable(out, gated_component_offenses(offenses, support)),
        )
        out[field] = published["index"]

    harm_counts = weighted_count(out, runtime.primary_vector)
    out[HARM_WEIGHTED_COUNT_COLUMN] = harm_counts
    if harm_index_is_supported(support):
        out[HARM_BURDEN_COLUMN] = count_first_index(
            counts=harm_counts,
            denominator=out[COMMON_DENOMINATOR_COLUMN],
            publishable=burden_publishable(
                out, gated_component_offenses(tuple(OFFENSES_7), support)
            ),
        )["index"]
    else:
        # Present and null at block group, the same shape the rare-offense publication policy
        # uses: the schema does not change with geography, and the claim is not made.
        out[HARM_BURDEN_COLUMN] = np.nan
    return out


def composite_normalizers(
    frame: pd.DataFrame,
    *,
    runtime: CompositeRuntime,
    support: str,
) -> dict[str, object]:
    """What a consumer needs to reproduce every published composite. A record, not a rederivation."""
    entries: dict[str, object] = {}
    for field, offenses in BURDEN_SPECS:
        published = count_first_index(
            counts=weighted_count(frame, {offense: 1.0 for offense in offenses}),
            denominator=frame[COMMON_DENOMINATOR_COLUMN],
            publishable=burden_publishable(frame, gated_component_offenses(offenses, support)),
        )
        entries[field] = {
            "offenses": list(offenses),
            "severity_weights": {offense: 1.0 for offense in offenses},
            "published_rows": int(pd.Series(published["publishable"]).sum()),
            "expected_count_total": float(published["count_total"]),
            "common_denominator_total": float(published["denominator_total"]),
            "reference_rate_per_100k": float(published["reference_rate_per_100k"]),
        }
    harm = harm_burden_index(
        frame,
        vector=runtime.primary_vector,
        publishable=burden_publishable(
            frame, gated_component_offenses(tuple(OFFENSES_7), support)
        ),
    )
    entries[HARM_BURDEN_COLUMN] = {
        "offenses": list(OFFENSES_7),
        "severity_vector_id": runtime.primary_vector_id,
        "severity_weights": runtime.primary_vector,
        "published_support": HARM_BURDEN_MIN_SUPPORT,
        "published_rows": (
            int(pd.Series(harm["publishable"]).sum()) if harm_index_is_supported(support) else 0
        ),
        "harm_weighted_count_total": float(harm["count_total"]),
        "common_denominator_total": float(harm["denominator_total"]),
        "reference_rate_per_100k": (
            float(harm["reference_rate_per_100k"]) if harm_index_is_supported(support) else float("nan")
        ),
    }
    return {
        "version": COMPOSITE_VERSION,
        "common_denominator_id": COMMON_DENOMINATOR_ID,
        "common_denominator_column": COMMON_DENOMINATOR_COLUMN,
        "semantics": COMMON_DENOMINATOR_SEMANTICS,
        "support": str(support),
        "fields": entries,
    }


def summarize_count_first_composites(*, runtime: CompositeRuntime) -> dict[str, object]:
    """The manifest block: which vector, from where, and what else it was tested against."""
    return {
        "enabled": True,
        "version": COMPOSITE_VERSION,
        "common_denominator_id": COMMON_DENOMINATOR_ID,
        "common_denominator_column": COMMON_DENOMINATOR_COLUMN,
        "semantics": COMMON_DENOMINATOR_SEMANTICS,
        "harm_published_support": HARM_BURDEN_MIN_SUPPORT,
        "severity_weights_path": str(runtime.weights_path),
        "primary_severity_vector": vector_metadata(runtime.weights, runtime.primary_vector_id),
        "alternative_severity_vectors": [
            vector_metadata(runtime.weights, vector_id) for vector_id in runtime.alternative_vector_ids
        ],
        "map_breaks": list(INDEX_MAP_BREAKS),
        "superseded_fields": dict(SUPERSEDED_COMPOSITE_FIELDS),
        "renamed_fields": dict(RENAMED_COMPOSITE_FIELDS),
        "contract": "analysis_scratch/final_phase/COMPOSITES_CONTRACT.md",
        "ags_values_used": False,
    }
