from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimerisk import allocation
from crimerisk.paths import RepoPaths

ROOT = Path(__file__).resolve().parents[1]


def _covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bg_id": ["010010000001", "010010000002", "010010000003"],
            "population": [100.0, 300.0, 0.0],
            "exposure_proxy_2024": [9.0, 9.0, 9.0],
            "opportunity_normalizer_larceny": [50.0, 50.0, 100.0],
            "opportunity_normalizer_robbery": [1.0, 1.0, 1.0],
        }
    )


def _components() -> pd.DataFrame:
    rows = []
    for offense in ("larceny", "robbery"):
        for jurisdiction, jtype, active, counts, support in (
            ("01:city", "municipal", False, [8.0, 2.0, 0.0], [1.0, 1.0, 0.5]),
            ("01:feed", "municipal", True, [3.0, 3.0, 4.0], [1.0, 1.0, 1.0]),
            ("01:statewide_overlap_layer", "statewide_overlap_layer", False, [1.0, 1.0, 1.0], [np.nan] * 3),
        ):
            for bg, count, fraction in zip(_covariates()["bg_id"], counts, support):
                rows.append(
                    {
                        "state_fips": "01",
                        "bg_id": bg,
                        "tract_id": bg[:11],
                        "jurisdiction_id": jurisdiction,
                        "jurisdiction_type": jtype,
                        "offense": offense,
                        "component_count": count,
                        "component_share": 0.0,
                        "city_incident_posterior_active": active,
                        allocation.FINISHED_SHARE_BLEND_SUPPORT_COLUMN: fraction,
                    }
                )
    return pd.DataFrame(rows)


def test_blend_moves_only_model_only_jurisdiction_mass_of_blended_offenses() -> None:
    components = _components()
    weights = {offense: (1.0, 0.0, 0.0) for offense in allocation.OFFENSES_7}
    weights["larceny"] = (0.8, 0.1, 0.1)
    out = allocation._apply_finished_share_blend(
        components, _covariates(), weights=weights, enable_exposure_ensemble=True
    )
    assert allocation.FINISHED_SHARE_BLEND_SUPPORT_COLUMN not in out.columns
    city = out[out["jurisdiction_id"].eq("01:city") & out["offense"].eq("larceny")]
    # model (0.8, 0.2, 0), population (100, 300, 0)/400, exposure (50, 50, 50)/150.
    expected = 10.0 * (
        0.8 * np.array([0.8, 0.2, 0.0])
        + 0.1 * np.array([0.25, 0.75, 0.0])
        + 0.1 * np.array([1 / 3, 1 / 3, 1 / 3])
    )
    assert np.allclose(city["component_count"], expected)
    assert np.isclose(city["component_count"].sum(), 10.0)
    assert np.allclose(city["component_share"], expected / 10.0)
    untouched = ~(out["jurisdiction_id"].eq("01:city") & out["offense"].eq("larceny"))
    assert out.loc[untouched, "component_count"].tolist() == components.loc[untouched, "component_count"].tolist()


def test_no_blended_offense_is_a_no_op() -> None:
    components = _components()
    weights = {offense: (1.0, 0.0, 0.0) for offense in allocation.OFFENSES_7}
    out = allocation._apply_finished_share_blend(
        components, _covariates(), weights=weights, enable_exposure_ensemble=True
    )
    pd.testing.assert_frame_equal(
        out, components.drop(columns=[allocation.FINISHED_SHARE_BLEND_SUPPORT_COLUMN])
    )


def test_shipped_config_blends_larceny_only() -> None:
    weights = allocation.load_finished_share_blend_weights(
        allocation.finished_share_blend_weights_path(RepoPaths.from_repo_root(ROOT))
    )
    assert weights["larceny"] == (0.8, 0.1, 0.1)
    assert all(point == (1.0, 0.0, 0.0) for offense, point in weights.items() if offense != "larceny")


def test_config_must_be_convex(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "offense": list(allocation.OFFENSES_7),
            "weight_model": 1.0,
            "weight_population": 0.0,
            "weight_primary_exposure": 0.0,
        }
    )
    table.loc[0, "weight_population"] = 0.5
    path = tmp_path / "blend.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="convex"):
        allocation.load_finished_share_blend_weights(path)
