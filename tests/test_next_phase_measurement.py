from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "next_phase_measurement.py"
SPEC = importlib.util.spec_from_file_location("next_phase_measurement_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_bg_population_frame = MODULE._bg_population_frame
_build_output_count_prior = MODULE._build_output_count_prior


def test_bg_population_frame_uses_requested_surface_and_year(tmp_path):
    surface = tmp_path / "candidate.parquet"
    pd.DataFrame(
        {
            "block_group_geoid": ["010010201001"],
            "state_fips": ["01"],
            "tract_id": ["01001020100"],
            "population_2024": [111.0],
            "population_2025": [222.0],
        }
    ).to_parquet(surface, index=False)

    actual = _bg_population_frame(surface, year=2025)

    assert actual.loc[0, "bg_id"] == "010010201001"
    assert actual.loc[0, "county_fips"] == "01001"
    assert actual.loc[0, "population"] == 222.0


def test_output_count_prior_reads_expected_counts_not_removed_legacy_counts():
    output = pd.DataFrame(
        {
            "bg_id": ["010010201001"],
            "tract_id": ["01001020100"],
            "state_fips": ["01"],
            "expected_count_murder": [0.25],
            "expected_count_robbery": [4.5],
        }
    )

    actual = _build_output_count_prior(output, offenses=["murder", "robbery", "burglary"])

    assert actual[["offense", "bg_weight"]].to_dict("records") == [
        {"offense": "murder", "bg_weight": 0.25},
        {"offense": "robbery", "bg_weight": 4.5},
    ]


def test_residual_holdout_refits_without_the_held_out_city(monkeypatch, tmp_path):
    cities = {"A": "01:municipal:place:1", "B": "02:municipal:place:2", "C": "03:municipal:place:3"}
    city_shares = pd.DataFrame(
        [
            {
                "city_name": name,
                "jurisdiction_id": jid,
                "state_fips": jid[:2],
                "year": 2025,
                "offense": "robbery",
                "block_group_geoid": f"{jid[:2]}000000000{index}",
                "incident_count": float(index + 1),
                "validation_case_type": "partial_year_municipal_validation_case" if name == "C" else "promoted_city_control",
            }
            for name, jid in cities.items()
            for index in range(2)
        ]
    )
    config = MODULE.AllocationBuildConfig(
        year=2025,
        residual_training_exclude_validation_case_types=("partial_year_municipal_validation_case",),
    )
    monkeypatch.setattr(MODULE, "resolve_allocation_build_config", lambda paths, *, config, _resolved=config: _resolved)

    def fake_prepare(*, city_shares, **_):
        frame = city_shares.rename(columns={"block_group_geoid": "bg_id"}).copy()
        frame["incident_total"] = frame.groupby("jurisdiction_id")["incident_count"].transform("sum")
        frame["true_share"] = frame["incident_count"] / frame["incident_total"]
        frame["model_share"] = 0.5
        frame["primary_geocode_quality_tier"] = "exact"
        frame["fallback_incident_share"] = 0.0
        frame["offense_volume_band"] = "low"
        frame["feature"] = 1.0
        return frame, ["feature"]

    trained_on: list[set[str]] = []

    def fake_fit(train, *, feature_cols, config):
        trained_on.append(set(train["jurisdiction_id"].astype(str)))
        return object()

    def fake_apply(test, *, fitted):
        out = test.copy()
        out["residual_model_share"] = out["true_share"]
        return out

    monkeypatch.setattr(MODULE, "prepare_city_residual_frame", fake_prepare)
    monkeypatch.setattr(MODULE, "fit_city_residual_model", fake_fit)
    monkeypatch.setattr(MODULE, "apply_city_residual_model", fake_apply)

    actual = MODULE.build_residual_holdout_diagnostics(
        paths=MODULE.RepoPaths.from_repo_root(tmp_path),
        year=2025,
        city_shares=city_shares,
        bg_prior=pd.DataFrame(),
        bg_crosswalk=pd.DataFrame(),
    )

    assert trained_on == [{cities["B"]}, {cities["A"]}]
    assert actual["holdout_jurisdiction_id"].tolist() == [cities["A"], cities["B"]]
    assert set(MODULE.RESIDUAL_HOLDOUT_COLUMNS).issubset(actual.columns)
    assert actual["residual_total_variation_distance"].tolist() == [0.0, 0.0]
    assert (actual["tvd_delta"] < 0).all()
    assert actual["validation_case_type"].eq("promoted_city_control").all()
