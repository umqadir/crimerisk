from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.model_surface import (
    ModelSurfaceConfig,
    _feature_group_for_column,
    _prepare_model_surface_context,
)
from crimerisk.paths import RepoPaths
from crimerisk.qa.model_diagnostics import feature_sanity_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--training-row-policy",
        choices=["observed_only", "include_estimated", "high_confidence_only"],
        default="observed_only",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "modeling" / "feature_sanity_2024.json")
    args = parser.parse_args()

    config = ModelSurfaceConfig(
        year=int(args.year),
        exclude_estimated_from_panel_from_training=args.training_row_policy != "include_estimated",
        high_confidence_training_only=args.training_row_policy == "high_confidence_only",
    )
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    controls, _, _, training, feature_cols, train_state_cols, base_x, _, _ = _prepare_model_surface_context(
        paths=paths,
        config=config,
    )

    diagnostics = feature_sanity_diagnostics(
        x=training[feature_cols],
        feature_group_resolver=_feature_group_for_column,
    )

    payload = {
        "year": int(args.year),
        "training_row_policy": str(args.training_row_policy),
        "selected_covariate_count": int(len(feature_cols)),
        "state_fixed_effect_count": int(len(train_state_cols)),
        "model_feature_count": int(base_x.shape[1]),
        "training_rows_total": int(len(training)),
        "feature_sanity": {
            "training_rows": diagnostics.training_rows,
            "feature_count": diagnostics.feature_count,
            "min_non_null_share": diagnostics.min_non_null_share,
            "p10_non_null_share": diagnostics.p10_non_null_share,
            "p50_non_null_share": diagnostics.p50_non_null_share,
            "p90_non_null_share": diagnostics.p90_non_null_share,
            "exact_or_near_complement_pairs": diagnostics.exact_or_near_complement_pairs,
            "strongest_positive_pairs": diagnostics.strongest_positive_pairs,
            "strongest_negative_pairs": diagnostics.strongest_negative_pairs,
            "high_abs_correlation_pairs": diagnostics.high_abs_correlation_pairs,
            "effective_rank": diagnostics.effective_rank,
            "condition_number": diagnostics.condition_number,
            "feature_group_counts": diagnostics.feature_group_counts,
            "low_coverage_features": diagnostics.low_coverage_features,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "training_row_policy": args.training_row_policy}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
