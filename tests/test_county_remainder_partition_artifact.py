"""The county-remainder evidence/prior audit columns survive to a stage-12 artifact."""

from __future__ import annotations

import pandas as pd
import pytest

import crimerisk.allocation as allocation
from crimerisk.allocation import (
    COUNTY_REMAINDER_PARTITION_AUDIT_COLUMNS,
    STATE_REMAINDER_TYPE,
    _build_county_remainder_group_targets,
    county_remainder_partition_path,
    write_county_remainder_partition,
)
from crimerisk.paths import RepoPaths

REMAINDER = "01:state_nonmunicipal_remainder"


def _targets(monkeypatch, tmp_path) -> pd.DataFrame:
    monkeypatch.setattr(allocation, "load_benchmark_imputation_units", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(
        allocation,
        "_load_crosswalk",
        lambda _paths: pd.DataFrame(
            {
                "ori": ["AL0010000", "AL0030000"],
                "state_fips": ["01", "01"],
                "jurisdiction_id": [REMAINDER, REMAINDER],
                "weight": [1.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        allocation,
        "_load_agency_master",
        lambda _paths: pd.DataFrame(
            {
                "ori9": ["AL0010000", "AL0030000"],
                "state_fips": ["01", "01"],
                "county_fips": ["001", "003"],
                "county_fips_source": ["srs_agency_header", "srs_agency_header"],
            }
        ),
    )
    # One observed larceny in county 001, none in 003: Pro's 1,000-over-1 example.
    monkeypatch.setattr(
        allocation,
        "_build_agency_risk_signals",
        lambda **_k: pd.DataFrame(
            {
                "ori9": ["AL0010000", "AL0030000"],
                "state_fips": ["01", "01"],
                "offense": ["larceny", "larceny"],
                "risk_signal_count": [1.0, 0.0],
            }
        ),
    )
    controls = pd.DataFrame(
        {
            "jurisdiction_type": [STATE_REMAINDER_TYPE],
            "state_fips": ["01"],
            "offense": ["larceny"],
            "adjusted_count_ags_core": [1000.0],
            "control_surface": ["smoothed"],
        }
    )
    estimates = pd.DataFrame(
        {
            "ori9": ["AL0010000", "AL0030000"],
            "state_fips": ["01", "01"],
            "offense": ["larceny", "larceny"],
            "reported_count_current_supported": [1.0, 0.0],
            "agency_adjustment_count": [0.0, 0.0],
        }
    )
    crosswalk = pd.DataFrame(
        {
            "state_fips": ["01", "01"],
            "block_group_geoid": ["010010001001", "010030001001"],
            "jurisdiction_type": [STATE_REMAINDER_TYPE] * 2,
            "allocation_share": [1.0, 1.0],
            "pop20": [100.0, 100.0],
        }
    )
    return _build_county_remainder_group_targets(
        paths=RepoPaths.from_repo_root(tmp_path),
        controls=controls,
        year=2025,
        bg_crosswalk=crosswalk,
        bg_prior=None,
        agency_estimates=estimates,
    )


def test_group_targets_carry_the_partition_audit_columns(monkeypatch, tmp_path):
    out = _targets(monkeypatch, tmp_path).set_index("group_id")

    assert set(COUNTY_REMAINDER_PARTITION_AUDIT_COLUMNS) <= set(out.columns)
    a = out.loc["01:state_nonmunicipal_remainder:county:01001"]
    assert a["target_count"] == pytest.approx(1000.0 * 1.25 / 1.5)
    assert a["remainder_evidence_mass"] == pytest.approx(1000.0 / 1.5)
    assert out["remainder_prior_mass"].sum() == pytest.approx(1000.0 * 0.5 / 1.5)
    assert out["target_count"].sum() == pytest.approx(1000.0)


def test_stage12_writes_the_partition_beside_the_manifest(monkeypatch, tmp_path):
    targets = _targets(monkeypatch, tmp_path)
    summary: dict[str, object] = {}

    path = write_county_remainder_partition(targets, tmp_path, year=2025, summary=summary)

    assert path == county_remainder_partition_path(tmp_path, year=2025)
    written = pd.read_parquet(path)
    pd.testing.assert_frame_equal(written, targets.reset_index(drop=True), check_dtype=False)
    assert summary["county_remainder_partition_rows"] == len(targets)
    assert summary["county_remainder_partition"]["remainder_prior_mass"] == pytest.approx(
        1000.0 * 0.5 / 1.5
    )
    assert write_county_remainder_partition(None, tmp_path, year=2025, summary={}) is None
