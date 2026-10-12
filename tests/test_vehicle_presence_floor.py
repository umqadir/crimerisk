"""D2 (2025.2): the published vehicle exposure is floored at a fixed ratio of daytime person presence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimerisk.denominators import MVT_VEHICLE_PRESENCE_FLOOR_RATIO, apply_vehicle_presence_floor


def test_floor_binds_only_where_vehicles_fall_below_the_presence_ratio():
    frame = pd.DataFrame(
        {
            "bg_id": ["060019821001", "060014001001", "000000000000"],
            # A campus: 31,670 present, 270 vehicles. An ordinary block group. An empty one.
            "exposure_proxy_2024": [31_670.0, 1_000.0, 0.0],
            "vehicle_exposure_2024": [270.0, 900.0, 0.0],
        }
    )
    out = apply_vehicle_presence_floor(frame)
    assert MVT_VEHICLE_PRESENCE_FLOOR_RATIO == 0.2511
    assert np.isclose(out.loc[0, "vehicle_exposure_2024"], 0.2511 * 31_670.0)
    assert out.loc[1, "vehicle_exposure_2024"] == 900.0
    assert out.loc[2, "vehicle_exposure_2024"] == 0.0
    # The input frame is not modified.
    assert frame.loc[0, "vehicle_exposure_2024"] == 270.0
