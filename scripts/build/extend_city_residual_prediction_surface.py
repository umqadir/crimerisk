"""Extend the promoted residual prediction surface to the whole current allocation frame.

Why this exists
---------------
The promoted surface is a frozen model product: for every component the allocator will
ever score it stores the model share it was fitted against and the log ratio the model
predicted. Output assembly consumes it instead of refitting, and it fails closed when a
component of the allocation frame is not in it.

A data refresh can add components. The 2025 Return A refresh gave agencies a target-year
report that had none before, which revived municipal jurisdictions the geometry had
written off as dead and moved block groups between the municipal and remainder lanes.
The allocation frame grew from 2,323,237 to 2,342,112 components, and 20,184 of them
carry model share the promoted surface was never scored on, across 15 jurisdictions in
nine states.

The repair is to score the new cells, not to drop them. The model is the same one the
surface was promoted from -- `scripts/build/reconstruct_city_residual_model.py`
reproduces it exactly from the frozen training inputs -- so extending the surface with
its predictions on the new cells leaves every existing row byte-identical and adds
predictions that come from the promoted model, not from a refit on new data.

How the frame is obtained
-------------------------
Not re-derived. The script runs the real output build and intercepts the surface
application, which is the one place the exact frame exists, writes the frame, and stops
the build there. Nothing is written to the candidate run, and `allocation.py` is not
modified: the interception replaces a module attribute for the duration of the call.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk import allocation as allocation_module  # noqa: E402
from crimerisk.city_residuals import (  # noqa: E402
    CITY_RESIDUAL_PREDICTION_COLUMNS,
    CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
    _load_city_residual_bg_features,
    apply_city_residual_model,
    apply_city_residual_prediction_surface,
    city_residual_fitted_model_path,
    city_residual_prediction_manifest_path,
    city_residual_prediction_surface_path,
    load_city_residual_fitted_model,
    load_city_residual_prediction_surface,
)
from crimerisk.paths import get_paths  # noqa: E402


class _FrameCaptured(RuntimeError):
    """Raised to stop the build once the allocation frame has been written."""


def capture_allocation_frame(*, out_path: Path, build_argv: list[str]) -> Path:
    """Run the real build until it applies the surface, and keep the frame it applies to."""
    from crimerisk import cli

    captured: dict[str, pd.DataFrame] = {}
    original = allocation_module.apply_city_residual_prediction_surface

    def intercept(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        captured["frame"] = frame.copy()
        raise _FrameCaptured

    allocation_module.apply_city_residual_prediction_surface = intercept
    argv = list(sys.argv)
    try:
        sys.argv = ["main.py", *build_argv]
        cli.main()
    except _FrameCaptured:
        pass
    finally:
        allocation_module.apply_city_residual_prediction_surface = original
        sys.argv = argv

    frame = captured.get("frame")
    if frame is None:
        raise SystemExit(
            "The build did not reach the residual surface application; nothing captured."
        )
    keep = [
        column
        for column in [*CITY_RESIDUAL_PREDICTION_KEY_COLUMNS, "model_share"]
        if column in frame.columns
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_parquet(out_path, index=False)
    print(f"captured allocation frame: {len(frame):,} rows -> {out_path}", flush=True)
    return out_path


def uncovered_components(
    *, frame: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """The components the allocator will ask about and the surface cannot answer.

    Zero-share components are left out: the application path already neutralises them,
    and inventing a prediction for a cell that carries no mass would be noise in the
    artifact.
    """
    work = frame.copy()
    work["bg_id"] = work["bg_id"].astype("string").str.zfill(12)
    work["state_fips"] = work["state_fips"].astype("string").str.zfill(2)
    work["jurisdiction_id"] = work["jurisdiction_id"].astype("string")
    work["offense"] = work["offense"].astype("string")
    work["model_share"] = pd.to_numeric(work["model_share"], errors="coerce").fillna(0.0)
    known = predictions[CITY_RESIDUAL_PREDICTION_KEY_COLUMNS].copy()
    known["_known"] = True
    merged = work.merge(known, on=CITY_RESIDUAL_PREDICTION_KEY_COLUMNS, how="left")
    missing = merged[merged["_known"].isna() & merged["model_share"].ne(0.0)]
    return missing.drop(columns="_known").drop_duplicates(
        CITY_RESIDUAL_PREDICTION_KEY_COLUMNS
    )


def score(*, paths, year: int, components: pd.DataFrame) -> pd.DataFrame:
    """Score new components with the promoted model, exactly as the verifier scores."""
    fitted, manifest = load_city_residual_fitted_model(
        city_residual_fitted_model_path(paths, year=int(year))
    )
    reproduction = manifest.get("reproduction") or {}
    if str(reproduction.get("status", "ok")) == "failed":
        raise SystemExit(
            "The reconstructed residual model is quarantined; refusing to extend the "
            "surface from it."
        )
    required = tuple(sorted(set(fitted.feature_cols) | set(fitted.burglary_feature_cols)))
    bg, _ = _load_city_residual_bg_features(
        paths=paths,
        year=int(year),
        extra_feature_paths=list(fitted.config.extra_feature_paths),
        feature_policy_path=fitted.config.feature_policy_path,
        exclude_feature_policy_classes=tuple(fitted.config.exclude_feature_policy_classes),
        exclude_feature_policy_classes_by_offense=fitted.config.exclude_feature_policy_classes_by_offense,
    )
    bg = bg[["bg_id", "state_fips", *required]]
    work = components.merge(bg, on=["bg_id", "state_fips"], how="left", validate="many_to_one")
    unscorable = work[list(required)].isna().all(axis=1)
    if bool(unscorable.any()):
        raise SystemExit(
            "New residual components have no block-group features and cannot be scored: "
            + str(
                work.loc[unscorable, list(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS)]
                .head(20)
                .to_dict("records")
            )
        )
    scored = apply_city_residual_model(work, fitted=fitted)
    out = scored[CITY_RESIDUAL_PREDICTION_KEY_COLUMNS].copy()
    out["model_share_at_fit"] = pd.to_numeric(
        scored["model_share"], errors="coerce"
    ).astype(float)
    out["predicted_log_ratio"] = pd.to_numeric(
        scored["predicted_log_ratio"], errors="coerce"
    ).astype(float)
    return out[CITY_RESIDUAL_PREDICTION_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend the promoted residual prediction surface to the current frame."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--frame",
        type=Path,
        default=Path("state/analysis/residual-model-reconstruction/allocation_frame_2025.parquet"),
        help="Where the captured allocation frame is written / read from.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Run the build up to the surface application to capture the frame first.",
    )
    parser.add_argument(
        "--build-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="The build-outputs arguments to run under interception, after --build-args.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = get_paths()
    year = int(args.year)
    frame_path = paths.repo_root / args.frame

    if args.capture:
        build_argv = args.build_args or [
            "build-outputs",
            "--year",
            str(year),
            "--control-surface",
            "smoothed",
        ]
        capture_allocation_frame(out_path=frame_path, build_argv=build_argv)
    if not frame_path.exists():
        raise SystemExit(f"No captured allocation frame at {frame_path}; pass --capture.")

    frame = pd.read_parquet(frame_path)
    surface_path = city_residual_prediction_surface_path(paths, year=year)
    predictions = load_city_residual_prediction_surface(surface_path)
    missing = uncovered_components(frame=frame, predictions=predictions)
    print(
        f"allocation frame {len(frame):,} rows; promoted surface {len(predictions):,} rows; "
        f"uncovered components carrying model share: {len(missing):,}",
        flush=True,
    )
    if missing.empty:
        print("surface already covers the allocation frame", flush=True)
        return 0
    print(
        "affected jurisdictions: "
        + str(sorted(set(missing["jurisdiction_id"].astype(str)))[:20]),
        flush=True,
    )
    scored = score(paths=paths, year=year, components=missing)
    if args.dry_run:
        print(scored.head(10).to_string(index=False))
        return 0

    extended = pd.concat([predictions, scored], ignore_index=True)
    if bool(extended.duplicated(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS).any()):
        raise SystemExit("Extending the surface produced duplicate component keys")
    # The promoted rows must come through untouched: this is an extension, not a refit.
    rejoined = predictions.merge(
        extended,
        on=CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
        how="left",
        suffixes=("_before", "_after"),
    )
    for column in ("model_share_at_fit", "predicted_log_ratio"):
        delta = (rejoined[f"{column}_before"] - rejoined[f"{column}_after"]).abs()
        if bool(delta.gt(0.0).any()):
            raise SystemExit(f"Extension changed a promoted {column}")

    backup = surface_path.with_suffix(surface_path.suffix + ".pre-extension")
    if not backup.exists():
        backup.write_bytes(surface_path.read_bytes())
    extended.to_parquet(surface_path, index=False)

    manifest_path = city_residual_prediction_manifest_path(surface_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extensions = list(manifest.get("frame_extensions") or [])
    extensions.append(
        {
            "year": year,
            "rows_added": int(len(scored)),
            "rows_before": int(len(predictions)),
            "rows_after": int(len(extended)),
            "jurisdictions": sorted(set(missing["jurisdiction_id"].astype(str))),
            "scored_from": str(city_residual_fitted_model_path(paths, year=year)),
            "reason": (
                "the data refresh added components the promoted surface had never been "
                "scored on; scored with the promoted model rather than dropped"
            ),
        }
    )
    manifest["frame_extensions"] = extensions
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Prove the extended surface answers the frame it was extended for. A model-share
    # drift is a different statement from a coverage gap and is not this script's to
    # fix: output assembly already handles it by re-scoring the frame with the
    # reconstructed model. Only an unanswered component is a failure here.
    reloaded = load_city_residual_prediction_surface(surface_path)
    still_missing = uncovered_components(frame=frame, predictions=reloaded)
    if not still_missing.empty:
        raise SystemExit(
            f"{len(still_missing):,} components remain uncovered after the extension"
        )
    try:
        apply_city_residual_prediction_surface(
            frame.assign(
                model_share=pd.to_numeric(frame["model_share"], errors="coerce").fillna(0.0)
            ),
            predictions=reloaded,
        )
        share_note = "model shares match the frame"
    except ValueError as exc:
        if not str(exc).startswith(
            "Promoted residual predictions do not match the current model-share input"
        ):
            raise
        share_note = (
            "model shares have drifted; output assembly will re-score the frame with "
            f"the reconstructed model ({exc})"
        )
    print(
        f"extended surface {len(predictions):,} -> {len(extended):,} rows; "
        f"covers the whole {len(frame):,}-row allocation frame; {share_note}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
