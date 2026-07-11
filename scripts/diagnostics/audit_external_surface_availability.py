from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOTS = [
    Path("data"),
    Path("state/modeling"),
    Path("state/reference"),
    Path("materials"),
    Path("docs"),
]
SURFACE_SUFFIXES = {".parquet", ".pq", ".csv", ".geojson", ".gpkg", ".shp"}
REFERENCE_SUFFIXES = {".pdf", ".txt", ".md", ".json"}
PRODUCT_TERMS = {
    "ags_crimerisk": ("ags", "crimerisk", "crime risk", "crime-index", "crime_index"),
    "cap_crimecast": ("cap", "crimecast", "crime cast"),
    "esri_crime_indexes": ("esri", "crime indexes", "crime-indexes", "crime_index"),
}
INTERNAL_PATH_PARTS = {
    "state/output",
    "state/tmp",
    "materials/tables/external_surface_benchmark_crimerisk",
    "materials/tables/dashboard_neighborhood_crimerisk",
    "materials/report",
    "materials/materials",
    "src/",
    ".venv",
}
INTERNAL_NAME_PARTS = {
    "crimerisk_block_group_2024_ags_core",
    "crimerisk_tract_2024_ags_core",
    "external_surface_benchmark_crimerisk_self_check",
    "external_surface_benchmark_crimerisk_tract_self_check",
}
PUBLIC_SOURCE_NOTES = [
    {
        "source": "Applied Geographic Solutions CrimeRisk",
        "url": "https://appliedgeographic.com/crimerisk/",
        "availability": "product/methodology page; no public national output surface found",
    },
    {
        "source": "CAP Index CRIMECAST data",
        "url": "https://capindex.com/solutions/data/",
        "availability": "commercial product page and sample reports; no public national output surface found",
    },
    {
        "source": "Esri Crime Indexes documentation",
        "url": "https://doc.arcgis.com/en/esri-demographics/latest/esri-demographics/crime-indexes.htm",
        "availability": "documentation says AGS crime indexes are available in Esri products; no exported surface file present locally",
    },
]


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _product_matches(text: str) -> list[str]:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    matches: list[str] = []
    for product, terms in PRODUCT_TERMS.items():
        if any(term in lowered for term in terms):
            matches.append(product)
    return sorted(set(matches))


def _is_internal(path_text: str) -> bool:
    lowered = path_text.lower()
    return any(part in lowered for part in INTERNAL_PATH_PARTS) or any(
        name in lowered for name in INTERNAL_NAME_PARTS
    )


def build_availability_audit(*, repo_root: Path, search_roots: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in search_roots:
        abs_root = root if root.is_absolute() else repo_root / root
        if not abs_root.exists():
            continue
        for path in abs_root.rglob("*"):
            if not path.is_file():
                continue
            rel_path = _rel(path, repo_root)
            product_matches = _product_matches(rel_path)
            if not product_matches:
                continue
            suffix = path.suffix.lower()
            is_surface_suffix = suffix in SURFACE_SUFFIXES
            is_reference_suffix = suffix in REFERENCE_SUFFIXES
            internal = _is_internal(rel_path)
            rows.append(
                {
                    "path": rel_path,
                    "suffix": suffix,
                    "size_bytes": int(path.stat().st_size),
                    "product_matches": ",".join(product_matches),
                    "is_surface_file_type": bool(is_surface_suffix),
                    "is_reference_file_type": bool(is_reference_suffix),
                    "is_internal_or_self_check": bool(internal),
                    "usable_external_surface_candidate": bool(is_surface_suffix and not internal),
                    "classification": (
                        "usable_external_surface_candidate"
                        if is_surface_suffix and not internal
                        else ("reference_or_methodology" if is_reference_suffix else "internal_or_non_surface")
                    ),
                }
            )
    table = pd.DataFrame(rows).sort_values(["usable_external_surface_candidate", "path"], ascending=[False, True])
    usable = table[table["usable_external_surface_candidate"]] if not table.empty else pd.DataFrame()
    reference = table[table["classification"].eq("reference_or_methodology")] if not table.empty else pd.DataFrame()
    summary = {
        "search_roots": [_rel((root if root.is_absolute() else repo_root / root), repo_root) for root in search_roots],
        "candidate_rows": int(len(table)),
        "usable_external_surface_count": int(len(usable)),
        "reference_or_methodology_count": int(len(reference)),
        "usable_external_surface_paths": usable["path"].tolist() if not usable.empty else [],
        "external_comparison_harness": "scripts/diagnostics/benchmark_external_surface.py",
        "harness_scoring_target": "observed incident shares in state/modeling/next_phase_validation_city_incident_share_surface_2024.parquet",
        "status": "external_surface_available" if len(usable) else "external_surface_unavailable",
        "public_source_notes": PUBLIC_SOURCE_NOTES,
    }
    return table, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether licensed/exported AGS/CAP/Esri crime-risk surfaces are locally available."
    )
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "materials" / "tables" / "external_surface_availability.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "state" / "modeling" / "external_surface_availability_2024.json",
    )
    args = parser.parse_args()
    roots = args.search_root or DEFAULT_SEARCH_ROOTS
    table, summary = build_availability_audit(repo_root=REPO_ROOT, search_roots=roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
