from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.city_residuals import (
    CITY_RESIDUAL_PREDICTION_COLUMNS,
    CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
    city_residual_prediction_surface_path,
    load_city_residual_prediction_surface,
)
from crimerisk.paths import RepoPaths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote the residual predictions from a validated allocation candidate."
    )
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    out_path = args.out or city_residual_prediction_surface_path(paths, year=int(args.year))
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace promoted residual predictions without --force: {out_path}")

    source_columns = [
        *CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
        "model_share",
        "city_residual_predicted_log_ratio",
    ]
    source = pd.read_parquet(args.candidate_audit, columns=source_columns)
    source = source.rename(
        columns={
            "model_share": "model_share_at_fit",
            "city_residual_predicted_log_ratio": "predicted_log_ratio",
        }
    )[CITY_RESIDUAL_PREDICTION_COLUMNS]
    # The component audit also contains overlap layers appended after the residual model
    # application.  They correctly have no model share or prediction and are not members of
    # the promoted model surface.
    source = source[
        source["model_share_at_fit"].notna() & source["predicted_log_ratio"].notna()
    ].copy()
    source["bg_id"] = source["bg_id"].astype("string").str.zfill(12)
    source["state_fips"] = source["state_fips"].astype("string").str.zfill(2)
    source["jurisdiction_id"] = source["jurisdiction_id"].astype("string")
    source["offense"] = source["offense"].astype("string")
    if bool(source[CITY_RESIDUAL_PREDICTION_KEY_COLUMNS].isna().any().any()):
        raise ValueError("Candidate residual predictions have null component keys")
    if bool(source.duplicated(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS).any()):
        raise ValueError("Candidate residual predictions have duplicate component keys")
    numeric = source[["model_share_at_fit", "predicted_log_ratio"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise ValueError("Candidate residual predictions contain non-finite values")
    source[["model_share_at_fit", "predicted_log_ratio"]] = numeric
    source = source.sort_values(CITY_RESIDUAL_PREDICTION_KEY_COLUMNS, kind="mergesort")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_parquet(out_path, index=False, compression="zstd")
    loaded = load_city_residual_prediction_surface(out_path)
    if len(loaded) != len(source):
        raise ValueError("Written residual prediction surface failed row-count verification")

    source_manifest_path = args.candidate_audit.parent / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            f"Candidate manifest is required to preserve residual feature-policy provenance: {source_manifest_path}"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    feature_policy_application = source_manifest.get("summary", {}).get(
        "city_residual_feature_policy"
    )
    if not isinstance(feature_policy_application, dict) or not feature_policy_application:
        raise ValueError(
            "Candidate manifest does not contain a residual feature-policy application summary"
        )

    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest = {
        "year": int(args.year),
        "source_run_id": str(args.source_run_id),
        "source_candidate_audit": str(args.candidate_audit.resolve()),
        "source_candidate_audit_sha256": _sha256(args.candidate_audit),
        "output": str(out_path.resolve()),
        "output_sha256": _sha256(out_path),
        "rows": int(len(source)),
        "key_columns": CITY_RESIDUAL_PREDICTION_KEY_COLUMNS,
        "prediction_column": "predicted_log_ratio",
        "input_binding_column": "model_share_at_fit",
        "feature_policy_application": feature_policy_application,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
