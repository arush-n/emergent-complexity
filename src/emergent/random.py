"""Compatibility exports for the compact ``emergent.random`` API."""

from .core.random import (
    generate_random_grids,
    key_from_seed,
    random_grid,
    random_grid_from_seed,
    random_grid_jit,
    random_rule,
    random_rules,
)

__all__ = [
    "generate_random_grids",
    "key_from_seed",
    "random_grid",
    "random_grid_from_seed",
    "random_grid_jit",
    "random_rule",
    "random_rules",
]
