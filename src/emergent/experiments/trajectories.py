"""Named trajectory wrappers for experiment code."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.grid import Grid
from ..core.rules import Rule
from ..core.simulate import generate_batched_trajectory as _generate_batched_trajectory
from ..core.simulate import generate_rule_trajectories as _generate_rule_trajectories
from ..core.simulate import generate_trajectory as _generate_trajectory


def generate_trajectories(initial_grid: Grid, rule: Rule, steps: int) -> Grid:
    """Generate a single-grid trajectory including the initial frame."""

    return _generate_trajectory(initial_grid, rule, steps)


def generate_batched_trajectories(initial_grids: Grid, rule: Rule, steps: int) -> Grid:
    """Generate trajectories in canonical ``(batch, time, height, width)`` order."""

    return _generate_batched_trajectory(initial_grids, rule, steps)


def generate_rule_trajectories(
    rules: Sequence[Rule],
    initial_grids: Grid,
    steps: int,
) -> Grid:
    """Generate ``(rule, batch, time, height, width)`` trajectories."""

    return _generate_rule_trajectories(rules, initial_grids, steps)
