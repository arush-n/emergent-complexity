"""Small binary patterns for demos and correctness tests."""

from __future__ import annotations

import jax.numpy as jnp

from ..core.grid import Grid

BLOCK = jnp.asarray(
    [
        [1, 1],
        [1, 1],
    ],
    dtype=jnp.uint8,
)

# A horizontal blinker; one Conway step rotates it vertically.
BLINKER = jnp.asarray([[1, 1, 1]], dtype=jnp.uint8)

GLIDER = jnp.asarray(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ],
    dtype=jnp.uint8,
)

PATTERNS: dict[str, Grid] = {
    "block": BLOCK,
    "blinker": BLINKER,
    "glider": GLIDER,
}


def get_pattern(name: str) -> Grid:
    """Return a named demo pattern or raise ``KeyError``."""

    try:
        return PATTERNS[name.lower()]
    except (AttributeError, KeyError) as exc:
        available = ", ".join(sorted(PATTERNS))
        raise KeyError(f"unknown pattern {name!r}; choose from {available}") from exc


def pattern_from_text(text: str) -> Grid:
    """Parse ``#``/``1`` as live and ``.``/``0`` as dead cells."""

    if not isinstance(text, str):
        raise TypeError("pattern text must be a string")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("pattern text must contain at least one row")
    width = len(lines[0])
    if width == 0 or any(len(line) != width for line in lines):
        raise ValueError("pattern rows must have equal non-zero widths")
    values: list[list[int]] = []
    for line in lines:
        row: list[int] = []
        for character in line:
            if character in "#1":
                row.append(1)
            elif character in ".0":
                row.append(0)
            else:
                raise ValueError("patterns may contain only '#', '1', '.', or '0'")
        values.append(row)
    return jnp.asarray(values, dtype=jnp.uint8)


def place_pattern(grid: Grid, pattern: Grid | str, row: int, col: int) -> Grid:
    """Return ``grid`` with ``pattern`` inserted at its top-left coordinate.

    The input grid is not mutated. Patterns must fit completely inside the
    target grid; use explicit coordinates so placement is reproducible.
    """

    target = jnp.asarray(grid)
    values = pattern_from_text(pattern) if isinstance(pattern, str) else jnp.asarray(pattern)
    if target.ndim != 2 or values.ndim != 2:
        raise ValueError("grid and pattern must both be two-dimensional")
    if not isinstance(row, int) or not isinstance(col, int):
        raise TypeError("row and col must be integers")
    outside = (
        row < 0
        or col < 0
        or row + values.shape[0] > target.shape[0]
        or col + values.shape[1] > target.shape[1]
    )
    if outside:
        raise ValueError("pattern must fit completely inside the target grid")
    binary_pattern = (values != 0).astype(target.dtype)
    return target.at[row : row + values.shape[0], col : col + values.shape[1]].set(binary_pattern)
