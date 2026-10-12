"""One national reference per offense and measure, frozen before any publication mask."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crimerisk.allocation import _count_derived_rate_index, national_reference_rate, surface_reference_rates


def _bg() -> pd.DataFrame:
    # Selma's 010479567021 is the whole of tract 01047956702; the other tract has two BGs, one
    # of them too small to publish.
    return pd.DataFrame(
        {
            "block_group_geoid": ["010479567021", "010479568001", "010479568002"],
            "tract_id": ["01047956702", "01047956800", "01047956800"],
            "count": [17.307, 40.0, 3.0],
            "population": [1_556.0, 3_000.0, 120.0],
        }
    )


def test_a_one_block_group_tract_has_its_block_groups_index():
    bg = _bg()
    publishable_bg = bg["population"].ge(550.0)
    reference = national_reference_rate(bg["count"], bg["population"])
    bg_index = _count_derived_rate_index(
        counts=bg["count"], denominator=bg["population"], publishable=publishable_bg, reference_rate_per_100k=reference
    )["index"]
    tract = bg.groupby("tract_id", as_index=False)[["count", "population"]].sum()
    # The tract surface publishes every tract, so a publishable-subset reference would differ.
    tract_index = _count_derived_rate_index(
        counts=tract["count"], denominator=tract["population"], publishable=tract["population"].ge(550.0),
        reference_rate_per_100k=reference,
    )["index"]
    assert tract_index.iloc[0] == pytest.approx(bg_index.iloc[0], rel=1e-12)
    # The tract's own full-surface reference is the same scalar.
    assert national_reference_rate(tract["count"], tract["population"]) == pytest.approx(reference, rel=1e-12)


def test_masking_a_cell_never_changes_another_cells_index():
    bg = _bg()
    everything = bg["population"].gt(0.0)
    full = _count_derived_rate_index(counts=bg["count"], denominator=bg["population"], publishable=everything)
    masked = _count_derived_rate_index(
        counts=bg["count"], denominator=bg["population"], publishable=everything & bg["population"].ge(550.0)
    )
    assert masked["national_rate_per_100k"] == pytest.approx(full["national_rate_per_100k"])
    assert masked["index"].iloc[0] == pytest.approx(full["index"].iloc[0])
    assert np.isnan(masked["index"].iloc[2])


def test_surface_reference_rates_reads_one_constant_per_offense_and_measure():
    surface = pd.DataFrame(
        {"primary_national_rate_per_100k_larceny": [1_097.1, 1_097.1], "resident_national_rate_per_100k_larceny": [1_100.0, 1_100.0]}
    )
    assert surface_reference_rates(surface) == {"primary:larceny": 1_097.1, "resident:larceny": 1_100.0}
    surface.loc[1, "primary_national_rate_per_100k_larceny"] = 1_134.9
    with pytest.raises(ValueError, match="not one constant"):
        surface_reference_rates(surface)


def test_the_resident_floor_is_one_per_special_use_type():
    from crimerisk.allocation import resident_rate_floor
    from crimerisk.composites import burden_publishable, COMMON_DENOMINATOR_COLUMN

    types = pd.Series(["ordinary", "park_open_space", "ordinary", None])
    assert list(resident_rate_floor(types, types.index)) == [300.0, 550.0, 300.0, 550.0]
    frame = pd.DataFrame(
        {
            COMMON_DENOMINATOR_COLUMN: [420.0, 420.0],
            "special_use_type": ["ordinary", "campus"],
            "special_use_resident_rate_allowed": [True, True],
        }
    )
    assert list(burden_publishable(frame)) == [True, False]
