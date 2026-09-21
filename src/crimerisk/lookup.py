"""The static lookup: one GEOID in, its headline published values out, with no server.

An edition's lookup is a directory of plain JSON files. There is **no API process, no database and
no query language** -- a client that knows a GEOID computes the shard filename from that GEOID
alone and fetches exactly one file:

    shard = geoid[: manifest.geographies[geography].prefix_length]
    GET   lookup/<geography>/<shard>.json
    row   = payload["rows"][geoid]          # values in payload["fields"] order

That is O(1) in the number of published rows, cacheable by any CDN or static host, and it survives
the edition being copied to a different root: nothing in a shard points at a mutable path.

THE SHARD KEY IS A GEOID PREFIX, and its LENGTH IS MEASURED, NOT ASSUMED. The prefix length for a
geography is the shortest one under which **every** shard fits the byte budget, which is what keeps
a single fetch small enough to be worth doing. A 5-digit (county) prefix is the natural key and is
what the coarse geographies land on, but it does not fit at block-group support: Los Angeles County
alone holds 6,591 block groups, an order of magnitude past the budget, so block groups shard on a
longer prefix and tracts on a longer one than that. The chosen length is recorded per geography in
`lookup/manifest.json`, so the client derives the filename from the manifest rather than from a
constant compiled into it.

VALUES ARE THE PUBLISHED VALUES, BIT FOR BIT. Nothing here rounds, rescales or re-derives: a float
is emitted through Python's shortest round-tripping repr, so `json.loads` of a shard returns the
same double the Parquet column holds. The lookup is a *transport* of the published surface, and a
value that differed from the table it was cut from would be a second, contradictory publication.

The published fields are the headline ones -- the seven per-offence indexes, the count-first
composites and the two relative scores, the per-offence reliability tier and the special-use type,
plus the unit's own name where the geography has one. Everything else stays in the tables: the
lookup exists so a reader can ask "what does this place look like" in one request, not to be a
second copy of a 792-column surface. Fields absent from a given support (a county rollup carries no
reliability tier) are recorded as absent for that geography rather than emitted null.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from crimerisk.composites import COUNT_FIRST_AGGREGATE_INDEX_FIELDS
from crimerisk.crime import OFFENSES_7
from crimerisk.rollups import ZCTA_ANTI_ZIP_CAVEAT


LOOKUP_VERSION = "static_lookup_v1"
LOOKUP_DIRNAME = "lookup"
MANIFEST_FILENAME = "manifest.json"
CHECKSUMS_FILENAME = "checksums.json"

# One shard is one HTTP request on a cold cache. 200 KB is the budget a single-place lookup is
# allowed to cost; the prefix length is whatever makes every shard fit it.
SHARD_BYTE_BUDGET = 200_000

# Compact separators, and the byte accounting below assumes them.
JSON_SEPARATORS = (",", ":")

# The client pattern, stated in the manifest so a reader never has to infer it from the file tree.
CLIENT_PATTERN = (
    "shard = geoid[:prefix_length] for the geography; GET lookup/<geography>/<shard>.json; "
    "payload['rows'][<full geoid>] is the value array, in payload['fields'] order. One fetch, no "
    "server, no index scan."
)


# --- the headline field list -------------------------------------------------------------------


def per_offense_index_fields() -> list[str]:
    return [f"index_{offense}_primary" for offense in OFFENSES_7]


def composite_fields() -> list[str]:
    """The count-first composites, including the two renamed index-average relative scores.

    Read off `composites.COUNT_FIRST_AGGREGATE_INDEX_FIELDS` rather than transcribed, so the lookup
    publishes whatever the composite lane publishes.
    """
    return list(COUNT_FIRST_AGGREGATE_INDEX_FIELDS)


def tier_fields() -> list[str]:
    return [f"reliability_tier_{offense}" for offense in OFFENSES_7]


def type_fields() -> list[str]:
    return ["special_use_type"]


HEADLINE_FIELD_FAMILIES: dict[str, list[str]] = {
    "per_offense_index": per_offense_index_fields(),
    "composite": composite_fields(),
    "tier": tier_fields(),
    "type": type_fields(),
}


def headline_fields() -> list[str]:
    """Every headline field, in publication order, before any surface is consulted.

    Deduplicated and asserted unique: the field list is the row's positional schema, so a repeated
    name would silently shift every value after it.
    """
    ordered: list[str] = []
    for family in ("per_offense_index", "composite", "tier", "type"):
        for field in HEADLINE_FIELD_FAMILIES[family]:
            if field not in ordered:
                ordered.append(field)
    return ordered


@dataclass(frozen=True)
class LookupGeography:
    """One published table the lookup indexes: what its GEOID column is, and what names it."""

    key: str
    id_column: str
    name_column: str | None = None
    caveat: str | None = None


LOOKUP_GEOGRAPHIES: tuple[LookupGeography, ...] = (
    LookupGeography(key="block_group", id_column="block_group_geoid"),
    LookupGeography(key="tract", id_column="tract_id"),
    LookupGeography(key="county", id_column="county_geoid", name_column="county_name"),
    LookupGeography(key="cbsa", id_column="cbsa_code", name_column="cbsa_title"),
    LookupGeography(key="zcta", id_column="zcta5", caveat=ZCTA_ANTI_ZIP_CAVEAT),
    LookupGeography(key="state", id_column="state_fips", name_column="state_abbr"),
)
LOOKUP_GEOGRAPHIES_BY_KEY: dict[str, LookupGeography] = {
    geo.key: geo for geo in LOOKUP_GEOGRAPHIES
}


def lookup_geography(key: str) -> LookupGeography:
    try:
        return LOOKUP_GEOGRAPHIES_BY_KEY[str(key)]
    except KeyError as exc:
        raise KeyError(
            f"{key!r} is not a lookup geography; known: {sorted(LOOKUP_GEOGRAPHIES_BY_KEY)}"
        ) from exc


def published_fields(geo: LookupGeography, columns: set[str]) -> tuple[list[str], list[str]]:
    """`(fields present on this surface, headline fields it does not carry)`, in order.

    The name column leads where the geography has one -- a lookup keyed on a code is close to
    useless without the label that code means -- and everything after it is a headline value.
    """
    ordered = [geo.name_column] if geo.name_column else []
    present: list[str] = [field for field in ordered if field in columns]
    absent: list[str] = []
    for field in headline_fields():
        (present if field in columns else absent).append(field)
    return present, absent


# --- serialization ------------------------------------------------------------------------------


def _python_values(series: pd.Series) -> list[object]:
    """One Parquet column as JSON-ready Python values, with no change of value.

    Floats go through `float()` and are emitted by `json.dumps` via the shortest representation
    that round-trips, so the value a client parses is the double the column holds. Missing is
    `None`, never a sentinel and never NaN (which is not JSON at all).
    """
    if pd.api.types.is_bool_dtype(series):
        return [None if value is None or pd.isna(value) else bool(value) for value in series]
    if pd.api.types.is_numeric_dtype(series):
        return [None if pd.isna(value) else float(value) for value in series]
    return [None if value is None or pd.isna(value) else str(value) for value in series]


def _row_fragments(frame: pd.DataFrame, fields: list[str]) -> list[str]:
    if len(set(fields)) != len(fields):
        raise ValueError(f"the lookup field list repeats a name: {fields}")
    columns = [_python_values(frame[field]) for field in fields]
    for field, values in zip(fields, columns):
        if len(values) != len(frame):
            raise ValueError(
                f"lookup column {field!r} yielded {len(values)} values for {len(frame)} rows"
            )
    fragments = [
        json.dumps(list(row), separators=JSON_SEPARATORS, allow_nan=False)
        for row in zip(*columns)
    ]
    if len(fragments) != len(frame):
        raise ValueError(
            f"lookup serialization produced {len(fragments)} rows for a {len(frame)}-row table"
        )
    return fragments


def _shard_header(
    *,
    edition_id: str,
    geo: LookupGeography,
    prefix_length: int,
    fields: list[str],
    shard: str,
) -> dict[str, object]:
    header: dict[str, object] = {
        "lookup_version": LOOKUP_VERSION,
        "edition_id": str(edition_id),
        "geography": geo.key,
        "id_field": geo.id_column,
        "prefix_length": int(prefix_length),
        "shard": shard,
        "fields": list(fields),
    }
    if geo.caveat is not None:
        # Emitted verbatim in every shard: a ZCTA value fetched on its own still carries the
        # caveat that must travel with it.
        header["caveat"] = geo.caveat
    return header


def _shard_text(header: dict[str, object], entries: list[tuple[str, str]]) -> str:
    """`{...header..., "rows": {"<geoid>": [...], ...}}`, assembled so its size is predictable."""
    head = json.dumps(header, separators=JSON_SEPARATORS, sort_keys=True)
    body = ",".join(
        f"{json.dumps(key, separators=JSON_SEPARATORS)}:{fragment}" for key, fragment in entries
    )
    return f"{head[:-1]},\"rows\":{{{body}}}}}\n"


def _entry_cost(key: str, fragment: str) -> int:
    return len(json.dumps(key, separators=JSON_SEPARATORS)) + 1 + len(fragment)


def _shard_bytes(head_length: int, costs: list[int]) -> int:
    # head without its closing brace, `,"rows":{`, the entries and their separators, `}}` and \n.
    return (head_length - 1) + 9 + sum(costs) + max(len(costs) - 1, 0) + 3


def choose_prefix_length(
    ids: list[str], costs: list[int], *, head_length: int, budget: int = SHARD_BYTE_BUDGET
) -> int:
    """The shortest GEOID prefix under which every shard fits the byte budget.

    Measured on the payload that will actually be written, so the answer is a property of this
    edition's values rather than a guess carried over from another one.
    """
    if not ids:
        return 1
    id_length = max(len(value) for value in ids)
    for prefix_length in range(1, id_length + 1):
        totals: dict[str, list[int]] = {}
        for value, cost in zip(ids, costs):
            totals.setdefault(value[:prefix_length], []).append(cost)
        if all(_shard_bytes(head_length, shard) <= budget for shard in totals.values()):
            return prefix_length
    return id_length


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the build ------------------------------------------------------------------------------------


def build_geography_shards(
    *,
    edition_id: str,
    geo: LookupGeography,
    surface_path: Path,
    out_dir: Path,
    budget: int = SHARD_BYTE_BUDGET,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Write one geography's shards; return its manifest entry and its per-shard checksums."""
    schema = pq.ParquetFile(str(surface_path)).schema_arrow
    columns = set(schema.names)
    if geo.id_column not in columns:
        raise KeyError(
            f"{surface_path.name} carries no {geo.id_column!r} column; the lookup cannot key "
            f"{geo.key} rows without their GEOID"
        )
    fields, absent = published_fields(geo, columns)
    frame = pd.read_parquet(surface_path, columns=[geo.id_column, *fields])
    ids = frame[geo.id_column].astype("string").fillna("").tolist()
    if any(value == "" for value in ids):
        raise ValueError(f"{surface_path.name} carries a row with no {geo.id_column}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{surface_path.name} carries duplicate {geo.id_column} values")

    fragments = _row_fragments(frame, fields)
    costs = [_entry_cost(key, fragment) for key, fragment in zip(ids, fragments)]
    # The header length is constant for a geography once the prefix length is fixed, so it is
    # measured once against a representative shard key.
    probe_length = len(
        json.dumps(
            _shard_header(
                edition_id=edition_id,
                geo=geo,
                prefix_length=len(ids[0]),
                fields=fields,
                shard=ids[0],
            ),
            separators=JSON_SEPARATORS,
            sort_keys=True,
        )
    )
    prefix_length = choose_prefix_length(ids, costs, head_length=probe_length, budget=budget)

    grouped: dict[str, list[tuple[str, str]]] = {}
    for key, fragment in zip(ids, fragments):
        grouped.setdefault(key[:prefix_length], []).append((key, fragment))

    geography_dir = out_dir / geo.key
    geography_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, dict[str, object]] = {}
    max_bytes = 0
    total_bytes = 0
    for shard in sorted(grouped):
        header = _shard_header(
            edition_id=edition_id,
            geo=geo,
            prefix_length=prefix_length,
            fields=fields,
            shard=shard,
        )
        text = _shard_text(header, sorted(grouped[shard]))
        size = len(text.encode("utf-8"))
        if size > budget:
            raise ValueError(
                f"lookup shard {geo.key}/{shard}.json is {size} bytes, over the {budget}-byte "
                "budget the prefix length was chosen to respect"
            )
        (geography_dir / f"{shard}.json").write_text(text)
        checksums[f"{geo.key}/{shard}.json"] = {
            "size_bytes": size,
            "sha256": sha256_text(text),
        }
        max_bytes = max(max_bytes, size)
        total_bytes += size

    entry: dict[str, object] = {
        "id_field": geo.id_column,
        "prefix_length": int(prefix_length),
        "path_template": f"{geo.key}/{{shard}}.json",
        "shards": len(grouped),
        "rows": int(len(ids)),
        "fields": fields,
        "absent_headline_fields": absent,
        "max_shard_bytes": int(max_bytes),
        "total_bytes": int(total_bytes),
        "source_table": surface_path.name,
    }
    if geo.caveat is not None:
        entry["caveat"] = geo.caveat
    return entry, checksums


def build_lookup_index(
    *,
    edition_root: Path,
    edition_id: str,
    tables: dict[str, Path],
    year: int = 2024,
    budget: int = SHARD_BYTE_BUDGET,
    built_at_utc: str | None = None,
) -> dict[str, object]:
    """Build the whole lookup under `<edition_root>/lookup/` and write its two manifests."""
    out_dir = Path(edition_root) / LOOKUP_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    geographies: dict[str, object] = {}
    checksums: dict[str, dict[str, object]] = {}
    for geo in LOOKUP_GEOGRAPHIES:
        path = tables.get(geo.key)
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"lookup input for {geo.key} is absent: {path}")
        entry, shard_checksums = build_geography_shards(
            edition_id=edition_id, geo=geo, surface_path=path, out_dir=out_dir, budget=budget
        )
        geographies[geo.key] = entry
        checksums.update(shard_checksums)

    checksums_payload = {
        "lookup_version": LOOKUP_VERSION,
        "edition_id": str(edition_id),
        "shards": checksums,
    }
    checksums_text = json.dumps(checksums_payload, indent=2, sort_keys=True) + "\n"
    (out_dir / CHECKSUMS_FILENAME).write_text(checksums_text)

    manifest: dict[str, object] = {
        "lookup_version": LOOKUP_VERSION,
        "edition_id": str(edition_id),
        "target_year": int(year),
        "built_at_utc": built_at_utc or datetime.now(timezone.utc).isoformat(),
        "shard_byte_budget": int(budget),
        "client_pattern": CLIENT_PATTERN,
        "headline_field_families": {
            family: list(fields) for family, fields in HEADLINE_FIELD_FAMILIES.items()
        },
        "value_policy": (
            "Values are the published values unchanged: floats are emitted with the shortest "
            "representation that round-trips to the same double, missing is null."
        ),
        "geographies": geographies,
        "checksums": {
            "path": CHECKSUMS_FILENAME,
            "sha256": sha256_text(checksums_text),
            "shards": len(checksums),
        },
        "totals": {
            "geographies": len(geographies),
            "shards": sum(int(entry["shards"]) for entry in geographies.values()),
            "rows": sum(int(entry["rows"]) for entry in geographies.values()),
            "bytes": sum(int(entry["total_bytes"]) for entry in geographies.values()),
            "max_shard_bytes": max(
                (int(entry["max_shard_bytes"]) for entry in geographies.values()), default=0
            ),
        },
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# --- the read side (the client pattern, in one function) ----------------------------------------


def shard_path(manifest: dict, *, geography: str, geoid: str) -> Path:
    """Where a GEOID's values live, derived from the GEOID and the manifest alone."""
    entry = (manifest.get("geographies") or {})[geography]
    prefix_length = int(entry["prefix_length"])
    return Path(geography) / f"{str(geoid)[:prefix_length]}.json"


def lookup_geoid(lookup_dir: Path, *, geography: str, geoid: str) -> dict[str, object] | None:
    """One fetch: read the shard the GEOID names and return its field->value mapping."""
    manifest = json.loads((Path(lookup_dir) / MANIFEST_FILENAME).read_text())
    path = Path(lookup_dir) / shard_path(manifest, geography=geography, geoid=geoid)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    values = (payload.get("rows") or {}).get(str(geoid))
    if values is None:
        return None
    return dict(zip(payload["fields"], values))
