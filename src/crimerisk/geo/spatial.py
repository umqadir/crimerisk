from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

from crimerisk.geo.tiger import TigerPolygon


@dataclass(frozen=True)
class PointMatch:
    polygon: TigerPolygon | None
    match_count: int


def assign_points_to_polygons(points: Sequence[Point], polygons: Sequence[TigerPolygon]) -> list[PointMatch]:
    if not polygons:
        return [PointMatch(polygon=None, match_count=0) for _ in points]

    geometries = [p.geometry for p in polygons]
    tree = STRtree(geometries)
    out: list[PointMatch] = []
    for pt in points:
        candidate_idx = tree.query(pt)
        candidates: list[TigerPolygon] = []
        for i in candidate_idx:
            poly = polygons[int(i)]
            if poly.geometry.contains(pt):
                candidates.append(poly)
        if not candidates:
            out.append(PointMatch(polygon=None, match_count=0))
            continue

        # If multiple polygons contain the point (rare), prefer the largest area
        # as a proxy for the broader governing boundary.
        best = max(candidates, key=lambda p: p.area)
        out.append(PointMatch(polygon=best, match_count=len(candidates)))
    return out

