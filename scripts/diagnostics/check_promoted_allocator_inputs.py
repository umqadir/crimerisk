#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crimerisk.allocation import promoted_next_phase_allocator_preflight  # noqa: E402
from crimerisk.paths import RepoPaths  # noqa: E402


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the promoted next-phase residual allocator inputs are present."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the preflight summary.",
    )
    args = parser.parse_args()

    paths = RepoPaths.from_repo_root(REPO_ROOT)
    summary = promoted_next_phase_allocator_preflight(paths, year=int(args.year))
    safe_summary = _json_safe(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(safe_summary, indent=2, sort_keys=True))
    print(json.dumps(safe_summary, indent=2, sort_keys=True))
    return 0 if bool(summary.get("ready")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
