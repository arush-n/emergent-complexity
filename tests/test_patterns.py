import numpy as np
import pytest

from emergent.core.grid import empty_grid
from emergent.io.patterns import BLOCK, GLIDER, get_pattern, pattern_from_text, place_pattern


def test_named_patterns_and_text_parser() -> None:
    assert get_pattern("block") is BLOCK
    np.testing.assert_array_equal(pattern_from_text(".#.\n..#\n###"), GLIDER)


def test_place_pattern_returns_new_grid() -> None:
    grid = empty_grid(6, 6)
    placed = place_pattern(grid, BLOCK, 2, 3)
    assert int(grid.sum()) == 0
    assert int(placed.sum()) == 4
    np.testing.assert_array_equal(placed[2:4, 3:5], BLOCK)


def test_pattern_must_fit() -> None:
    with pytest.raises(ValueError):
        place_pattern(empty_grid(3, 3), GLIDER, 1, 1)
