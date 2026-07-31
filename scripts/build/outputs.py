from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.paths import RepoPaths
from crimerisk.crime import OFFENSES_7
from crimerisk.allocation import (
    AllocationBuildConfig,
    DEFAULT_EB_ALPHA_BY_OFFENSE,
    DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES,
    DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH,
    DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR,
    write_v2_outputs,
)


def _parse_eb_alpha(values: list[str]) -> tuple[tuple[str, float], ...]:
    # Start from the allocation default (per-offense EB prior strength) so a no-flag build picks it
    # up; explicit --eb-alpha still overrides per offense.
    alpha = {offense: a for offense, a in DEFAULT_EB_ALPHA_BY_OFFENSE}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--eb-alpha values must use offense=value")
        offense, raw_alpha = value.split("=", 1)
        offense = offense.strip()
        if offense not in alpha:
            raise argparse.ArgumentTypeError(f"unknown offense in --eb-alpha: {offense!r}")
        parsed = float(raw_alpha)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--eb-alpha values must be positive")
        alpha[offense] = parsed
    return tuple((offense, alpha[offense]) for offense in OFFENSES_7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bg-ags-core-out",
        type=Path,
        default=REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_ags_core.parquet",
    )
    parser.add_argument(
        "--tract-ags-core-out",
        type=Path,
        default=REPO_ROOT / "state" / "output" / "crimerisk_tract_2024_ags_core.parquet",
    )
    parser.add_argument(
        "--bg-fbi-out",
        type=Path,
        default=REPO_ROOT / "state" / "output" / "crimerisk_block_group_2024_fbi_calibrated.parquet",
    )
    parser.add_argument(
        "--tract-fbi-out",
        type=Path,
        default=REPO_ROOT / "state" / "output" / "crimerisk_tract_2024_fbi_calibrated.parquet",
    )
    parser.add_argument(
        "--emit-fbi-calibrated",
        action="store_true",
        help="Also write the optional fbi_calibrated derivative outputs.",
    )
    parser.add_argument(
        "--no-county-anchoring",
        action="store_false",
        dest="enable_county_anchoring",
        help=(
            "Disable allocation-local county anchoring for state nonmunicipal remainder "
            "and county-evidenced overlap rows."
        ),
    )
    parser.add_argument(
        "--build-manifest-out",
        type=Path,
        default=None,
        help=(
            "Optional JSON sidecar recording the resolved allocator configuration, key input file "
            "stats, and output file stats."
        ),
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--force-controls-rebuild",
        action="store_true",
        help="Rebuild jurisdiction controls and jurisdiction_year_estimates before allocating outputs.",
    )
    parser.add_argument(
        "--force-reporting-regimes-rebuild",
        action="store_true",
        help="When rebuilding controls from build-outputs, also rebuild agency_year_reporting_regimes.parquet.",
    )
    parser.add_argument(
        "--force-geometry-rebuild",
        action="store_true",
        help="Rebuild block and block-group jurisdiction crosswalks before allocating outputs.",
    )
    parser.add_argument(
        "--force-bg-prior-rebuild",
        action="store_true",
        help="Rebuild the cached bg_prior_long modeling surface before allocating outputs.",
    )
    parser.add_argument(
        "--bg-prior-path",
        type=Path,
        default=None,
        help=(
            "Optional cached bg_prior_long parquet to read or rebuild. Defaults to the production "
            "arm-B prior path when the model-surface config matches arm B."
        ),
    )
    parser.add_argument(
        "--model-surface-prior-anchor",
        type=str,
        default=DEFAULT_MODEL_SURFACE_PRIOR_ANCHOR,
        choices=("resident_population", "offense_denominator"),
        help="Model-surface prior anchor used when generating bg_prior_long.",
    )
    parser.add_argument(
        "--model-surface-feature-policy-path",
        type=Path,
        default=DEFAULT_MODEL_SURFACE_FEATURE_POLICY_PATH,
        help="Feature transfer policy parquet used by the production model surface.",
    )
    parser.add_argument(
        "--model-surface-exclude-feature-policy-class",
        action="append",
        default=list(DEFAULT_MODEL_SURFACE_EXCLUDE_FEATURE_POLICY_CLASSES),
        help=(
            "Feature-transfer final_class to exclude from the production model surface. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--force-city-incident-share-rebuild",
        action="store_true",
        help="Rebuild city_incident_share_surface.parquet before allocating outputs.",
    )
    parser.add_argument(
        "--force-city-incident-source-refresh",
        action="store_true",
        help="Refresh raw city-source caches before rebuilding city_incident_share_surface.parquet.",
    )
    parser.add_argument(
        "--residual-training-city-shares-path",
        type=Path,
        default=None,
        help=(
            "Optional city-share surface used only to train the residual allocator. "
            "Direct city incident overrides still come from city_incident_share_surface.parquet."
        ),
    )
    parser.add_argument(
        "--no-promoted-next-phase-allocator",
        action="store_false",
        dest="use_promoted_next_phase_allocator",
        help=(
            "Disable automatic use of the promoted next-phase residual allocator inputs "
            "when those artifacts are present."
        ),
    )
    parser.add_argument(
        "--residual-training-exclude-validation-case-type",
        action="append",
        default=[],
        help=(
            "validation_case_type value to exclude from the optional residual-training surface. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--residual-training-extra-bg-features-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional BG feature parquet used only by the residual allocator model. "
            "May be repeated. It does not alter the jurisdiction-total model surface."
        ),
    )
    parser.add_argument(
        "--eb-alpha",
        action="append",
        default=[],
        metavar="OFFENSE=ALPHA",
        help=(
            "Override the empirical-Bayes alpha for one offense. May be repeated. "
            "Default is 1.0 for each seven-index offense."
        ),
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    summary = write_v2_outputs(
        paths=paths,
        block_group_ags_core_out=args.bg_ags_core_out,
        tract_ags_core_out=args.tract_ags_core_out,
        block_group_fbi_out=args.bg_fbi_out if args.emit_fbi_calibrated else None,
        tract_fbi_out=args.tract_fbi_out if args.emit_fbi_calibrated else None,
        build_manifest_out=args.build_manifest_out,
        config=AllocationBuildConfig(
            year=int(args.year),
            force_controls_rebuild=bool(args.force_controls_rebuild),
            force_reporting_regimes_rebuild=bool(args.force_reporting_regimes_rebuild),
            force_geometry_rebuild=bool(args.force_geometry_rebuild),
            force_bg_prior_rebuild=bool(args.force_bg_prior_rebuild),
            bg_prior_path=args.bg_prior_path,
            model_surface_prior_anchor=str(args.model_surface_prior_anchor),
            model_surface_feature_policy_path=args.model_surface_feature_policy_path,
            model_surface_exclude_feature_policy_classes=tuple(
                str(value) for value in args.model_surface_exclude_feature_policy_class
            ),
            force_city_incident_share_rebuild=bool(args.force_city_incident_share_rebuild),
            force_city_incident_source_refresh=bool(args.force_city_incident_source_refresh),
            use_promoted_next_phase_allocator=bool(args.use_promoted_next_phase_allocator),
            residual_training_city_shares_path=args.residual_training_city_shares_path,
            residual_training_exclude_validation_case_types=tuple(
                str(value) for value in args.residual_training_exclude_validation_case_type
            ),
            residual_training_extra_bg_feature_paths=tuple(args.residual_training_extra_bg_features_path),
            eb_alpha_by_offense=_parse_eb_alpha([str(value) for value in args.eb_alpha]),
            enable_county_anchoring=bool(args.enable_county_anchoring),
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {args.bg_ags_core_out}")
    print(f"Wrote {args.tract_ags_core_out}")
    if args.emit_fbi_calibrated:
        print(f"Wrote {args.bg_fbi_out}")
        print(f"Wrote {args.tract_fbi_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
