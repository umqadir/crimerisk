"""'Direct' provenance follows the city posterior's actual prior/direct mixture."""

from __future__ import annotations

import pandas as pd
import pytest

from crimerisk.confidence import _component_source_long


def _component(active: bool, prior_fraction: float, direct_share: float, prior_share: float, count: float) -> dict:
    posterior = (1.0 - prior_fraction) * direct_share + prior_fraction * prior_share
    return {
        "state_fips": "08", "bg_id": "080310001001", "tract_id": "08031000100", "jurisdiction_id": "08:municipal:place:0820000",
        "offense": "murder", "component_count_after": count, "component_share": 1.0, "model_share": prior_share,
        "city_incident_posterior_active": active, "city_posterior_prior_fraction": prior_fraction,
        "city_posterior_direct_share": direct_share, "city_posterior_share": posterior,
    }


def test_a_posterior_that_is_almost_all_prior_is_not_labelled_direct():
    """Denver murder: prior fraction 0.99999705; every public tract used to say direct_city_incident."""
    frame = pd.DataFrame([_component(True, 0.99999705, 0.02, 0.01, 0.7)])
    out = _component_source_long(frame, geo_col="bg_id").iloc[0]
    assert out["source_mode"] == "modeled_transfer"
    assert out["_source_direct_share"] < 1e-4


def test_the_direct_share_is_the_mixture_weighted_part_of_the_cell():
    frame = pd.DataFrame([_component(True, 0.25, 0.04, 0.02, 3.0), _component(False, float("nan"), 0.0, 0.0, 1.0)])
    out = _component_source_long(frame, geo_col="bg_id").iloc[0]
    direct_fraction = 0.75 * 0.04 / (0.75 * 0.04 + 0.25 * 0.02)
    assert out["_source_direct_share"] == pytest.approx(3.0 * direct_fraction / 4.0)
