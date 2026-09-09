"""Sampling helpers for future experiments; no behavior classification is done here."""

from __future__ import annotations

from typing import Any

import jax

from ..core.random import generate_random_grids, random_rules
from ..core.rules import Rule


def sample_initial_grids(
    key: Any,
    count: int,
    height: int,
    width: int,
    density: float,
):
    """Sample ``count`` reproducible initial grids as ``(batch, height, width)``."""

    if not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    return generate_random_grids(jax.random.split(key, count), height, width, density)


def sample_rules(key: Any, count: int) -> list[Rule]:
    """Sample random rules without attaching any behavioral labels."""

    return random_rules(key, count)
