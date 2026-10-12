"""Accounting tests for candidate surface comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "compare_candidate_surfaces.py"


def _module():
    spec = importlib.util.spec_from_file_location("compare_candidate_surfaces_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPARE = _module()


def test_transfer_net_excludes_explicit_terminal_unlocated_mass():
    ledger = pd.DataFrame(
        [
            {
                "service_scope_id": "scope",
                "canonical_target_ori": "AA0000001",
                "offense": "robbery",
                "source_state_fips": "04",
                "allocation_state_fips": "04",
                "source_target_count": 100.0,
                "allocation_count": 60.0,
                "route_reason": "",
            },
            {
                "service_scope_id": "scope",
                "canonical_target_ori": "AA0000001",
                "offense": "robbery",
                "source_state_fips": "04",
                "allocation_state_fips": "35",
                "source_target_count": 100.0,
                "allocation_count": 30.0,
                "route_reason": "",
            },
            {
                "service_scope_id": "scope",
                "canonical_target_ori": "AA0000001",
                "offense": "robbery",
                "source_state_fips": "04",
                "allocation_state_fips": pd.NA,
                "source_target_count": 100.0,
                "allocation_count": 10.0,
                "route_reason": "service_no_eligible_receiver",
            },
        ]
    )
    net = COMPARE._transfer_net_by_state(ledger)
    assert net.loc[("04", "robbery")] == pytest.approx(-30.0)
    assert net.loc[("35", "robbery")] == pytest.approx(30.0)
    assert net.sum() == pytest.approx(0.0)


def test_control_change_requires_hash_bound_manifest_input(tmp_path: Path):
    control_input = tmp_path / "scope.csv"
    control_input.write_text("scope\n")
    arm = {
        "control_change_contract": {
            "allowed": True,
            "input_path": str(control_input.resolve()),
            "input_sha256": COMPARE._sha256(control_input),
            "reason": "scope population correction",
        }
    }
    assert COMPARE._control_change_authorized(
        arm, manifest_text=str(control_input.resolve())
    ) == (True, "scope population correction")

    with pytest.raises(ValueError, match="not bound"):
        COMPARE._control_change_authorized(arm, manifest_text="{}")


def test_control_changes_are_not_authorized_by_default():
    assert COMPARE._control_change_authorized({}, manifest_text="{}") == (False, None)


def test_controls_path_uses_verified_frozen_snapshot(tmp_path: Path):
    candidate = tmp_path / "candidate"
    frozen = candidate / "frozen_upstream"
    controls = frozen / "controls" / "jurisdiction_controls_smoothed_2025.parquet"
    controls.parent.mkdir(parents=True)
    controls.write_bytes(b"frozen-controls")
    (frozen / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "controls/jurisdiction_controls_smoothed_2025.parquet",
                        "sha256": COMPARE._sha256(controls),
                    }
                ]
            }
        )
    )
    manifest_path = candidate / "manifest.json"
    manifest = {
        "year": 2025,
        "input_file_stats": {
            "jurisdiction_controls_smoothed": {
                "exists": True,
                "size_bytes": controls.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    controls.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest))

    resolved, inventory = COMPARE._controls_path(
        {"name": "candidate", "manifest_path": str(manifest_path)}, manifest
    )
    assert resolved == controls
    assert inventory == frozen / "manifest.json"


def test_controls_path_rejects_corrupt_override_and_unbound_legacy(tmp_path: Path):
    candidate = tmp_path / "candidate"
    frozen = candidate / "frozen_upstream"
    controls = frozen / "controls" / "jurisdiction_controls_smoothed_2025.parquet"
    controls.parent.mkdir(parents=True)
    controls.write_bytes(b"frozen-controls")
    (frozen / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "controls/jurisdiction_controls_smoothed_2025.parquet",
                        "sha256": COMPARE._sha256(controls),
                    }
                ]
            }
        )
    )
    manifest_path = candidate / "manifest.json"
    manifest = {
        "year": 2025,
        "input_file_stats": {
            "jurisdiction_controls_smoothed": {
                "exists": True,
                "size_bytes": controls.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    controls.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        },
    }
    override = tmp_path / "wrong.parquet"
    override.write_bytes(b"wrong-controls")
    with pytest.raises(ValueError, match="does not match frozen inventory"):
        COMPARE._controls_path(
            {
                "name": "candidate",
                "manifest_path": str(manifest_path),
                "controls_path": str(override),
            },
            manifest,
        )

    legacy_manifest = tmp_path / "legacy.json"
    legacy_manifest.write_text("{}")
    with pytest.raises(ValueError, match="controls_sha256"):
        COMPARE._controls_path(
            {
                "name": "legacy",
                "manifest_path": str(legacy_manifest),
                "controls_path": str(controls),
            },
            manifest,
        )
