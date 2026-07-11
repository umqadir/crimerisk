from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths
from crimerisk.reference_layers import ReferenceLayerBuildConfig, write_v2_reference_layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build resolved reference-layer artifacts.")
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument(
        "--agency-master",
        type=Path,
        default=Path("state/reference/agency_master.parquet"),
    )
    parser.add_argument(
        "--agency-master-supplement-path",
        type=Path,
        default=Path("configs/agency_master_supplement.csv"),
    )
    parser.add_argument(
        "--provisional-local-path",
        type=Path,
        default=Path("state/reference/inputs/provisional_local_agency_matches.parquet"),
    )
    parser.add_argument(
        "--resolved-local-tail-path",
        type=Path,
        default=Path("state/reference/inputs/local_queue_resolved_final.parquet"),
    )
    parser.add_argument(
        "--nonlocal-final-path",
        type=Path,
        default=Path("state/reference/inputs/nonmunicipal_special_resolved_final.parquet"),
    )
    parser.add_argument(
        "--nonlocal-auto-path",
        type=Path,
        default=Path("state/reference/inputs/nonmunicipal_auto_defaults.parquet"),
    )
    parser.add_argument(
        "--municipal-override-path",
        type=Path,
        default=Path("configs/municipal_geometry_overrides.csv"),
    )
    parser.add_argument(
        "--local-override-path",
        type=Path,
        default=Path("configs/local_resolution_overrides.csv"),
    )
    parser.add_argument(
        "--full-local-out",
        type=Path,
        default=Path("state/reference/local_agency_resolved_full.parquet"),
    )
    parser.add_argument(
        "--full-nonlocal-out",
        type=Path,
        default=Path("state/reference/nonlocal_agency_resolved_full.parquet"),
    )
    parser.add_argument(
        "--jurisdiction-out",
        type=Path,
        default=Path("state/reference/jurisdiction_master.parquet"),
    )
    parser.add_argument(
        "--crosswalk-out",
        type=Path,
        default=Path("state/reference/agency_to_jurisdiction_crosswalk.parquet"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    summary = write_v2_reference_layers(
        paths=paths,
        full_local_out_path=paths.repo_root / args.full_local_out,
        full_nonlocal_out_path=paths.repo_root / args.full_nonlocal_out,
        jurisdiction_out_path=paths.repo_root / args.jurisdiction_out,
        crosswalk_out_path=paths.repo_root / args.crosswalk_out,
        config=ReferenceLayerBuildConfig(
            year_start=args.year_start,
            year_end=args.year_end,
            agency_master_path=paths.repo_root / args.agency_master,
            provisional_local_path=paths.repo_root / args.provisional_local_path,
            resolved_local_tail_path=paths.repo_root / args.resolved_local_tail_path,
            nonlocal_final_path=paths.repo_root / args.nonlocal_final_path,
            nonlocal_auto_path=paths.repo_root / args.nonlocal_auto_path,
            municipal_override_path=paths.repo_root / args.municipal_override_path,
            local_override_path=paths.repo_root / args.local_override_path,
            agency_master_supplement_path=paths.repo_root / args.agency_master_supplement_path,
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {paths.repo_root / args.full_local_out}")
    print(f"Wrote {paths.repo_root / args.full_nonlocal_out}")
    print(f"Wrote {paths.repo_root / args.jurisdiction_out}")
    print(f"Wrote {paths.repo_root / args.crosswalk_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
