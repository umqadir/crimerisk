"""The LAR rules of scripts/build/01_tribal_territory_footprints.py on synthetic geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

BUILDER = Path(__file__).resolve().parents[1] / "scripts" / "build" / "01_tribal_territory_footprints.py"
spec = importlib.util.spec_from_file_location("tribal_territory_builder", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

# Lon/lat boxes near Atmore AL, small enough to stay planar.
X0, Y0, D = -87.50, 31.10, 0.01


def box(i: float, j: float, w: float = 1.0, h: float = 1.0):
    return shapely.box(X0 + i * D, Y0 + j * D, X0 + (i + w) * D, Y0 + (j + h) * D)


def _lars(features: list[tuple[str, str, object]]):
    frame = gpd.GeoDataFrame(
        {"LARID": [f[0] for f in features], "LARNAME": [f[1] for f in features]},
        geometry=[f[2] for f in features],
        crs=4269,
    )
    lars = builder.LandAreaRepresentations.__new__(builder.LandAreaRepresentations)
    lars.frame = frame
    lars.equal_area = frame.to_crs(5070)
    lars.index = lars.equal_area.sindex
    return lars


def test_name_tokens_drop_generic_words():
    assert builder._name_tokens("Navajo (Ramah)") == {"navajo", "ramah"}
    assert builder._name_tokens("Coushatta Reservation") == {"coushatta"}


def test_lar_maps_by_overlap_share_or_name():
    census = box(0, 0, 4, 4)
    lars = _lars(
        [
            ("L1", "Inside Mostly", box(1, 1, 2, 2)),  # all of it inside the Census area
            ("L2", "Tampa", box(3.5, 0, 6, 1)),  # mostly outside, name matches
            ("L3", "Neighbour", box(3.9, 3.9, 6, 6)),  # sliver overlap, other tribe
            ("L4", "Far Away", box(20, 20)),  # no overlap
        ]
    )
    union, decisions = lars.mapped("X", census, "Tampa Reservation")
    mapped = {d["larid"]: d["mapped"] for d in decisions}
    assert mapped == {"L1": True, "L2": True, "L3": False}
    assert union.covers(box(5, 0.5, 0.1, 0.1))  # L2's land outside the Census area joins
    assert not union.intersects(box(5.5, 5.5, 0.1, 0.1))


class _Points:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)

    def inside(self, geom):
        hit = shapely.contains_xy(geom, self.xy[:, 0], self.xy[:, 1])
        return self.xy[hit, 0], self.xy[hit, 1]


def _state(tmp_path, blocks: list[tuple[str, object, int]]):
    frame = gpd.GeoDataFrame(
        {"GEOID20": [b[0] for b in blocks]}, geometry=[b[1] for b in blocks], crs=4269
    )
    path = tmp_path / "blocks.gpkg"
    frame.to_file(path, driver="GPKG")
    state = builder.StateBlocks.__new__(builder.StateBlocks)
    state.path = str(path)
    state.crs = "EPSG:4269"
    state.attrs = pd.DataFrame(
        {
            "pop": [b[2] for b in blocks],
            "housing": 0, "aland": 1, "wac": 0, "rac": 0, "retail": 0, "industrial": 0,
            "bg": [b[0][:12] for b in blocks],
        },
        index=pd.Index([b[0] for b in blocks], name="GEOID20"),
    )
    return state


def test_block_membership_rules(tmp_path):
    census = box(0, 0, 1, 1)
    lar = shapely.union_all([box(1, 0, 1, 1), box(2, 0, 1, 1), box(3, 0, 0.2, 1)])
    state = _state(
        tmp_path,
        [
            ("010010001001000", box(0, 0), 10),  # census
            ("010010001001001", box(1, 0), 0),  # unpopulated, wholly LAR
            ("010010001001002", box(2, 0), 50),  # populated, wholly LAR: left out
            ("010010001001003", box(3, 0), 0),  # 20% LAR with a destination point on LAR land
            ("010010001001004", box(4, 0), 0),  # outside
        ],
    )
    points = _Points([[X0 + 3.1 * D, Y0 + 0.5 * D], [X0 + 4.5 * D, Y0 + 0.5 * D]])
    blocks = state.territory_blocks(census, lar, points).set_index("GEOID20")["membership"]
    assert blocks.to_dict() == {
        "010010001001000": "census",
        "010010001001001": "lar_majority",
        "010010001001002": "excluded_populated_lar",
        "010010001001003": "lar_facility",
    }


def test_without_lar_only_census_blocks(tmp_path):
    state = _state(tmp_path, [("010010001001000", box(0, 0), 10), ("010010001001001", box(1, 0), 0)])
    blocks = state.territory_blocks(box(0, 0, 1, 1))
    assert blocks["membership"].tolist() == ["census"]


@pytest.mark.parametrize("inside, whole, expected", [(0.0, 0.0, np.nan), (2.0, 4.0, 0.5), (5.0, 4.0, 1.0)])
def test_share(inside, whole, expected):
    np.testing.assert_allclose(builder._share([inside], [whole]), [expected])


def test_facility_min_lar_fraction_option(tmp_path, monkeypatch):
    census = box(0, 0, 1, 1)
    lar = shapely.union_all([box(3, 0, 0.2, 1)])
    state = _state(tmp_path, [("010010001001000", box(0, 0), 10), ("010010001001003", box(3, 0), 0)])
    points = _Points([[X0 + 3.1 * D, Y0 + 0.5 * D]])
    assert "lar_facility" in set(state.territory_blocks(census, lar, points)["membership"])
    monkeypatch.setattr(builder.StateBlocks, "facility_min_lar_fraction", 0.3)
    assert "lar_facility" not in set(state.territory_blocks(census, lar, points)["membership"])
