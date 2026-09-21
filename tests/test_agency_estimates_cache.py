from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from crimerisk import allocation, controls
from crimerisk.build_freshness import dependency_stamp_path
from crimerisk.paths import RepoPaths
from crimerisk.trend_fills import AGENCY_TARGET_ESTIMATE_COLUMNS


def _paths(root: Path) -> RepoPaths:
    return RepoPaths.from_repo_root(root)


def _estimate_frame(*, count: float = 12.5, source: str = "masked_gap_history") -> pd.DataFrame:
    row = {column: pd.NA for column in AGENCY_TARGET_ESTIMATE_COLUMNS}
    row.update(
        {
            "ori9": "AA0000001",
            "state_fips": "01",
            "offense": "larceny",
            "reported_count_current": 8.0,
            "reported_count_current_supported": 8.0,
            "estimated_count": count,
            "agency_adjustment_count": count - 8.0,
            "agency_estimate_source": source,
            "partial_uplift_sanity_cap_flag": False,
            "agency_estimate_review_flag": False,
            "masked_gap_reclassified": True,
            "accepted_observed_mass": 8.0,
            "repair_mass": count - 8.0,
            "invalid_fragment_audit_mass": 0.0,
            "unresolved_mass": 0.0,
        }
    )
    return pd.DataFrame([row], columns=AGENCY_TARGET_ESTIMATE_COLUMNS)


def test_controls_authoritative_cache_is_reused_without_builder(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    dependency = tmp_path / "configs" / "population-and-scope.csv"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("v1\n")
    monkeypatch.setattr(
        allocation,
        "_agency_estimates_dependency_paths",
        lambda _paths, *, year: [dependency],
    )
    allocation._AGENCY_ESTIMATES_MEMO.clear()
    expected = _estimate_frame(count=14.25, source="controls_masked_gap_recovery")

    cache_path = allocation.persist_agency_allocation_target_estimates_cache(
        paths=paths,
        year=2025,
        agency_estimates=expected,
    )
    allocation._AGENCY_ESTIMATES_MEMO.clear()
    monkeypatch.setattr(
        allocation,
        "build_agency_allocation_target_estimates",
        lambda **_kwargs: pytest.fail("allocation builder must not run on a current cache"),
    )

    actual = allocation._build_agency_allocation_target_estimates(paths=paths, year=2025)

    assert cache_path.exists()
    assert dependency_stamp_path(cache_path).exists()
    assert actual.loc[0, "estimated_count"] == pytest.approx(14.25)
    assert actual.loc[0, "agency_estimate_source"] == "controls_masked_gap_recovery"
    assert actual[["ori9", "state_fips", "offense"]].to_dict("records") == expected[
        ["ori9", "state_fips", "offense"]
    ].to_dict("records")


def test_dependency_change_invalidates_disk_and_process_memo(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    population = tmp_path / "state" / "reference" / "jurisdiction_master.parquet"
    config = tmp_path / "configs" / "service_wide_agency_scopes.csv"
    population.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    population.write_text("population-v1\n")
    config.write_text("scope-v1\n")
    dependencies = [population, config]
    monkeypatch.setattr(
        allocation,
        "_agency_estimates_dependency_paths",
        lambda _paths, *, year: dependencies,
    )
    allocation._AGENCY_ESTIMATES_MEMO.clear()
    allocation.persist_agency_allocation_target_estimates_cache(
        paths=paths,
        year=2025,
        agency_estimates=_estimate_frame(count=10.0, source="controls_v1"),
    )
    population.write_text("population-v2\n")
    rebuilt = _estimate_frame(count=17.0, source="rebuilt_after_population_change")
    calls = []

    def build(**_kwargs):
        calls.append(True)
        return rebuilt.copy()

    monkeypatch.setattr(allocation, "build_agency_allocation_target_estimates", build)

    actual = allocation._build_agency_allocation_target_estimates(paths=paths, year=2025)

    assert calls == [True]
    assert actual.loc[0, "estimated_count"] == pytest.approx(17.0)
    assert actual.loc[0, "agency_estimate_source"] == "rebuilt_after_population_change"


def test_cache_writer_rejects_noncanonical_or_duplicate_frames(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="canonical column schema"):
        allocation.persist_agency_allocation_target_estimates_cache(
            paths=paths,
            year=2025,
            agency_estimates=_estimate_frame().drop(columns="agency_estimate_source"),
        )


def test_target_cache_dependencies_exclude_residual_allocation_implementation(
    tmp_path,
):
    paths = _paths(tmp_path)

    dependencies = allocation._agency_estimates_dependency_paths(paths, year=2025)

    assert Path(allocation.__file__).resolve() not in dependencies
    assert tmp_path / "src" / "crimerisk" / "trend_fills.py" in dependencies
    duplicate = pd.concat([_estimate_frame(), _estimate_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate agency-offense"):
        allocation.persist_agency_allocation_target_estimates_cache(
            paths=paths,
            year=2025,
            agency_estimates=duplicate,
        )


def test_controls_writer_publishes_final_estimates_for_allocation(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    (tmp_path / "state" / "controls").mkdir(parents=True)
    dependency = tmp_path / "configs" / "scope.csv"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("scope-v1\n")
    expected = _estimate_frame(count=19.5, source="controls_masked_gap_recovery")
    frame = pd.DataFrame({"value": [1.0]})
    benchmark = SimpleNamespace(mass_ledger=pd.DataFrame({"value": [1.0]}))
    monkeypatch.setattr(controls, "_ensure_controls_dependencies", lambda **_kwargs: None)
    monkeypatch.setattr(
        controls,
        "build_controls_bundle",
        lambda **_kwargs: (frame, frame, frame, frame, benchmark, expected),
    )
    monkeypatch.setattr(controls, "write_benchmark_imputation_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(controls, "write_jurisdiction_ownership_exclusions", lambda *args, **kwargs: None)
    monkeypatch.setattr(controls, "_controls_dependency_paths", lambda *args, **kwargs: [dependency])
    monkeypatch.setattr(
        allocation,
        "_agency_estimates_dependency_paths",
        lambda _paths, *, year: [dependency],
    )
    allocation._AGENCY_ESTIMATES_MEMO.clear()

    controls.write_v2_controls(
        paths=paths,
        state_out_path=tmp_path / "state" / "controls" / "state.parquet",
        jurisdiction_out_path=tmp_path / "state" / "controls" / "jurisdiction.parquet",
        jurisdiction_year_estimates_out_path=tmp_path
        / "state"
        / "controls"
        / "jurisdiction_year.parquet",
        config=controls.ControlBuildConfig(year=2025),
    )
    monkeypatch.setattr(
        allocation,
        "build_agency_allocation_target_estimates",
        lambda **_kwargs: pytest.fail("allocation builder must reuse controls cache"),
    )

    actual = allocation._build_agency_allocation_target_estimates(paths=paths, year=2025)

    pd.testing.assert_frame_equal(actual, expected)
