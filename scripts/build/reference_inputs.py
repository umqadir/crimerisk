from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import get_paths
from crimerisk.reference_layers import ReferenceInputPromotionConfig, promote_v2_reference_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote reviewed local-resolution outputs into the canonical V2 reference input surface."
    )
    parser.add_argument(
        "--provisional-local-source-path",
        type=Path,
        default=Path("state/review/queues/local_resolution/provisional_local_agency_matches.parquet"),
    )
    parser.add_argument(
        "--resolved-local-tail-source-path",
        type=Path,
        default=Path("state/review/queues/local_resolution/local_queue_resolved_final.parquet"),
    )
    parser.add_argument(
        "--nonlocal-final-source-path",
        type=Path,
        default=Path("state/review/queues/local_resolution/nonmunicipal_special_resolved_final.parquet"),
    )
    parser.add_argument(
        "--nonlocal-auto-source-path",
        type=Path,
        default=Path("state/review/queues/local_resolution/nonmunicipal_auto_defaults.parquet"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    summary = promote_v2_reference_inputs(
        paths=paths,
        config=ReferenceInputPromotionConfig(
            provisional_local_source_path=paths.repo_root / args.provisional_local_source_path,
            resolved_local_tail_source_path=paths.repo_root / args.resolved_local_tail_source_path,
            nonlocal_final_source_path=paths.repo_root / args.nonlocal_final_source_path,
            nonlocal_auto_source_path=paths.repo_root / args.nonlocal_auto_source_path,
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
