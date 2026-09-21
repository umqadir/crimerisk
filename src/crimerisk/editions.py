"""Versioned editions: what a release IS, on disk, and how one is assembled.

An edition is a frozen, self-describing directory: the published surfaces (as GeoParquet, geometry
joined), the exact rollups, the field dictionary, the build manifest and the validation summary
the build passed under, plus an `edition.json` that names the edition, its type, its coverage
universe, and the SHA-256 of every file it contains. Nothing in an edition points at a mutable
path: a reader holding the directory can recompute every published rate and index from the
published counts and denominators inside it.

Two edition TYPES exist, each with its own builder, and neither builder will build the other's
type -- asking for the wrong one fails with that sentence rather than producing an annual edition
under a provisional name or the reverse:

* `annual` -- Surface 1, annual accounting. Estimated reported offences for one calendar year,
  exact conservation to the jurisdiction totals, frozen once published. Built by
  `scripts/release/package_edition.py`.
* `quarterly-provisional` -- Surface 3, the provisional nowcast. Quarterly, from the FBI's
  preliminary/monthly releases, jurisdiction-level updates over stable shares, carrying as-of
  dates and a revision history and making no finality claim. Built by
  `scripts/release/build_provisional_edition.py` from `crimerisk.nowcast`. Its payload is a
  `provisional/` directory of factor, jurisdiction, state and block-group-overlay tables rather
  than the annual `rollups/` + `surfaces/` pair: a provisional edition re-scales the annual
  surface, it does not re-derive it, so it has no rollups of its own to publish.

Edition ids are `<year><sequence>-<type>`: `2024A-annual` is the first annual edition of target
year 2024, `2024B-annual` a later re-issue of the same year. A provisional edition's sequence is
its quarter (`2025Q3-quarterly-provisional`), which is why the sequence is a token rather than a
letter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd

from crimerisk.paths import RepoPaths


EDITION_LAYOUT_VERSION = "edition_v1"

EDITION_ID_PATTERN = re.compile(r"^(?P<year>\d{4})(?P<sequence>[A-Z]|Q[1-4])-(?P<type>[a-z-]+)$")


@dataclass(frozen=True)
class EditionType:
    key: str
    surface: str
    cadence: str
    estimand: str
    buildable: bool
    note: str


ANNUAL = EditionType(
    key="annual",
    surface="Surface 1 - annual accounting",
    cadence="annual",
    estimand="estimated reported offences, one calendar year",
    buildable=True,
    note=(
        "Exact conservation to the jurisdiction official totals. Frozen once published; a revision "
        "is a new edition sequence, never an edit."
    ),
)
QUARTERLY_PROVISIONAL = EditionType(
    key="quarterly-provisional",
    surface="Surface 3 - provisional nowcast",
    cadence="quarterly",
    estimand="provisional estimate of reported offences, as of a stated date",
    buildable=True,
    note=(
        "Built from the FBI preliminary/monthly release lane, stamped with that source's own "
        "as-of date and carrying a revision history. No finality claim is made by a provisional "
        "edition: it is revised continuously, it is not comparable to the annual accounting "
        "surface, and it never supersedes an annual one."
    ),
)

EDITION_TYPES: dict[str, EditionType] = {
    ANNUAL.key: ANNUAL,
    QUARTERLY_PROVISIONAL.key: QUARTERLY_PROVISIONAL,
}


class EditionTypeNotBuildable(RuntimeError):
    pass


class EditionTypeMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class EditionId:
    year: int
    sequence: str
    type_key: str

    def __str__(self) -> str:
        return f"{self.year}{self.sequence}-{self.type_key}"

    @property
    def edition_type(self) -> EditionType:
        return EDITION_TYPES[self.type_key]


def parse_edition_id(value: str) -> EditionId:
    match = EDITION_ID_PATTERN.match(str(value))
    if match is None:
        raise ValueError(
            f"{value!r} is not a valid edition id; expected <year><sequence>-<type>, "
            "e.g. 2024A-annual"
        )
    type_key = match.group("type")
    if type_key not in EDITION_TYPES:
        raise ValueError(
            f"{value!r} names unknown edition type {type_key!r}; known: {sorted(EDITION_TYPES)}"
        )
    return EditionId(
        year=int(match.group("year")), sequence=match.group("sequence"), type_key=type_key
    )


def require_buildable(edition_id: EditionId) -> None:
    edition_type = edition_id.edition_type
    if not edition_type.buildable:
        raise EditionTypeNotBuildable(
            f"edition type {edition_type.key!r} is defined but not buildable: {edition_type.note}"
        )


def require_edition_type(edition_id: EditionId, expected: str, *, builder: str) -> None:
    """Each builder builds exactly one type.

    The types differ in estimand, not only in file layout, so a mismatch is never a harmless
    naming slip: it would publish an annual accounting surface under a provisional id (implying a
    freshness it does not have) or a provisional nowcast under an annual id (implying a finality
    it explicitly disclaims).
    """
    if edition_id.type_key != expected:
        other = EDITION_TYPES[edition_id.type_key]
        raise EditionTypeMismatch(
            f"{builder} builds {expected!r} editions; {edition_id} is a "
            f"{edition_id.type_key!r} edition ({other.surface}). Build it with its own builder."
        )


# --- layout -------------------------------------------------------------------------------------


SURFACES_DIRNAME = "surfaces"
ROLLUPS_DIRNAME = "rollups"
PROVISIONAL_DIRNAME = "provisional"

ANNUAL_DIRECTORIES: tuple[str, ...] = (SURFACES_DIRNAME, ROLLUPS_DIRNAME)
PROVISIONAL_DIRECTORIES: tuple[str, ...] = (PROVISIONAL_DIRNAME,)


@dataclass(frozen=True)
class EditionLayout:
    root: Path

    @property
    def surfaces_dir(self) -> Path:
        return self.root / SURFACES_DIRNAME

    @property
    def rollups_dir(self) -> Path:
        return self.root / ROLLUPS_DIRNAME

    @property
    def provisional_dir(self) -> Path:
        return self.root / PROVISIONAL_DIRNAME

    @property
    def readme_path(self) -> Path:
        return self.root / "README.md"

    @property
    def dictionary_path(self) -> Path:
        return self.root / "FIELD_DICTIONARY.md"

    @property
    def manifest_path(self) -> Path:
        return self.root / "edition.json"

    @property
    def build_manifest_path(self) -> Path:
        return self.root / "build_manifest.json"

    @property
    def validation_summary_path(self) -> Path:
        return self.root / "validation_summary.json"

    @property
    def rollup_summary_path(self) -> Path:
        return self.rollups_dir / "rollup_summary.json"

    def prepare(self, *, directories: tuple[str, ...] = ANNUAL_DIRECTORIES) -> None:
        """Create the edition root and the payload directories its type actually publishes.

        A provisional edition has no `rollups/` and no `surfaces/`: it re-scales the annual
        surface rather than re-deriving it, and an empty directory named `rollups` in a
        provisional edition would advertise a rollup that was never computed.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        for name in directories:
            (self.root / name).mkdir(parents=True, exist_ok=True)


def editions_dir(paths: RepoPaths) -> Path:
    return paths.state_dir / "editions"


def edition_layout(paths: RepoPaths, edition_id: EditionId, *, root: Path | None = None) -> EditionLayout:
    base = root or editions_dir(paths)
    return EditionLayout(root=base / str(edition_id))


# --- GeoParquet ----------------------------------------------------------------------------------

# TIGER/Line 2020 is NAD83 (EPSG:4269). The surfaces are written in their source CRS rather than
# reprojected, so no coordinate in a published edition has been moved by the packaging step.
TIGER_CRS_EPSG = 4269
GEOMETRY_COLUMN = "geometry"

TIGER_SPECS: dict[str, dict[str, str]] = {
    "block_group": {
        "directory": "tiger_bg",
        "template": "tl_2020_{state_fips}_bg.zip",
        "geoid_field": "GEOID",
        "id_column": "block_group_geoid",
        "id_length": "12",
    },
    "tract": {
        "directory": "tiger_tracts",
        "template": "tl_2020_{state_fips}_tract.zip",
        "geoid_field": "GEOID",
        "id_column": "tract_id",
        "id_length": "11",
    },
}


def load_tiger_geometry(
    paths: RepoPaths, *, geography: str, state_fips: list[str]
) -> "pd.DataFrame":
    """Full-resolution 2020 TIGER geometry for one geography, one row per GEOID.

    Imported lazily: geopandas pulls GDAL bindings that the rest of the rollup lane does not need.
    """
    import geopandas as gpd

    spec = TIGER_SPECS[geography]
    directory = paths.data_dir / spec["directory"]
    frames = []
    for fips in sorted(set(state_fips)):
        path = directory / spec["template"].format(state_fips=str(fips).zfill(2))
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is absent; the edition cannot publish GeoParquet without the TIGER "
                f"{geography} file for state {fips}"
            )
        frame = gpd.read_file(f"zip://{path}", columns=[spec["geoid_field"]])
        frame = frame.rename(columns={spec["geoid_field"]: spec["id_column"]})
        frames.append(frame[[spec["id_column"], GEOMETRY_COLUMN]])
    out = pd.concat(frames, ignore_index=True)
    out[spec["id_column"]] = (
        out[spec["id_column"]].astype("string").str.zfill(int(spec["id_length"]))
    )
    return gpd.GeoDataFrame(out, geometry=GEOMETRY_COLUMN, crs=f"EPSG:{TIGER_CRS_EPSG}")


def write_geoparquet(
    *,
    paths: RepoPaths,
    surface_path: Path,
    geography: str,
    out_path: Path,
    compression: str = "zstd",
) -> dict[str, object]:
    """Join TIGER geometry onto a published surface and write it as GeoParquet.

    Every surface row must find a geometry: a published cell with no boundary is a defect in the
    geography lane, not something to drop quietly at packaging time.
    """
    import geopandas as gpd

    spec = TIGER_SPECS[geography]
    id_column = spec["id_column"]
    surface = pd.read_parquet(surface_path)
    surface[id_column] = surface[id_column].astype("string").str.zfill(int(spec["id_length"]))
    state_fips = sorted(surface[id_column].str.slice(0, 2).dropna().unique().tolist())

    geometry = load_tiger_geometry(paths, geography=geography, state_fips=state_fips)
    geometry = geometry.drop_duplicates(id_column)

    merged = surface.merge(geometry, on=id_column, how="left")
    missing = int(merged[GEOMETRY_COLUMN].isna().sum())
    if missing:
        examples = merged.loc[merged[GEOMETRY_COLUMN].isna(), id_column].head(5).tolist()
        raise ValueError(
            f"{missing} {geography} rows have no TIGER geometry (e.g. {examples}); "
            "the edition refuses to publish a geometry-less GeoParquet row"
        )
    gdf = gpd.GeoDataFrame(merged, geometry=GEOMETRY_COLUMN, crs=f"EPSG:{TIGER_CRS_EPSG}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False, compression=compression)
    return {
        "path": str(out_path),
        "geography": geography,
        "rows": int(len(gdf)),
        "columns": int(gdf.shape[1]),
        "crs": f"EPSG:{TIGER_CRS_EPSG}",
        "geometry_column": GEOMETRY_COLUMN,
        "geometry_source": f"TIGER/Line 2020 {spec['directory']}",
        "compression": compression,
    }


# --- assembly -------------------------------------------------------------------------------------


def sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def copy_into(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


@dataclass
class EditionManifest:
    edition_id: EditionId
    layout: EditionLayout
    input_dir: Path
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        edition_type = self.edition_id.edition_type
        return {
            "layout_version": EDITION_LAYOUT_VERSION,
            "edition_id": str(self.edition_id),
            "year": self.edition_id.year,
            "sequence": self.edition_id.sequence,
            "edition_type": {
                "key": edition_type.key,
                "surface": edition_type.surface,
                "cadence": edition_type.cadence,
                "estimand": edition_type.estimand,
                "buildable": edition_type.buildable,
                "note": edition_type.note,
            },
            "defined_edition_types": {
                key: {
                    "surface": value.surface,
                    "cadence": value.cadence,
                    "estimand": value.estimand,
                    "buildable": value.buildable,
                    "note": value.note,
                }
                for key, value in EDITION_TYPES.items()
            },
            "input_dir": str(self.input_dir),
            "created_at_utc": self.created_at_utc,
            **self.payload,
        }

    def write(self) -> Path:
        path = self.layout.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n")
        return path
