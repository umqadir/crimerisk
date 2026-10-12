from __future__ import annotations

from crimerisk.allocation import DEFAULT_RARE_OFFENSE_INFORMATION_CONSTANT_BY_OFFENSE
from crimerisk.model_surface import RARE_OFFENSE_INFORMATION_CONSTANT_GRID


def test_the_completed_grid_contains_the_old_boundary_as_an_interior_point() -> None:
    grid = RARE_OFFENSE_INFORMATION_CONSTANT_GRID
    assert grid == tuple(sorted(grid))
    assert grid[0] == 0.1
    assert grid[-1] == 1_000_000.0
    assert grid.index(100.0) > 0
    assert grid.index(100.0) < len(grid) - 1


def test_only_murder_moves_after_the_e4_excluded_reselection() -> None:
    constants = dict(DEFAULT_RARE_OFFENSE_INFORMATION_CONSTANT_BY_OFFENSE)
    assert constants == {"murder": 1.0, "rape": 100.0}
