import numpy as np

from emergent.core.grid import empty_grid
from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.simulate import run_steps
from emergent.core.step import neighbor_count, step, step_jit
from emergent.io.patterns import BLINKER, BLOCK, GLIDER, place_pattern


def conway_masks():
    return rule_to_masks(parse_rule("B3/S23"))


def test_block_is_stable() -> None:
    grid = place_pattern(empty_grid(4, 4), BLOCK, 1, 1)
    birth, survival = conway_masks()
    np.testing.assert_array_equal(step(grid, birth, survival), grid)
    np.testing.assert_array_equal(run_steps(grid, parse_rule("B3/S23"), 20), grid)


def test_blinker_oscillates() -> None:
    grid = place_pattern(empty_grid(5, 5), BLINKER, 2, 1)
    expected_vertical = np.zeros((5, 5), dtype=np.uint8)
    expected_vertical[1:4, 2] = 1
    birth, survival = conway_masks()
    np.testing.assert_array_equal(step(grid, birth, survival), expected_vertical)
    np.testing.assert_array_equal(step(expected_vertical, birth, survival), grid)


def test_glider_translates_after_four_generations() -> None:
    grid = place_pattern(empty_grid(10, 10), GLIDER, 1, 1)
    expected = place_pattern(empty_grid(10, 10), GLIDER, 2, 2)
    np.testing.assert_array_equal(run_steps(grid, parse_rule("B3/S23"), 4), expected)


def test_empty_grid_stays_empty_and_neighbor_count_wraps() -> None:
    empty = empty_grid(6, 6)
    birth, survival = conway_masks()
    np.testing.assert_array_equal(step_jit(empty, birth, survival), empty)

    corner = empty.at[0, 0].set(1)
    counts = neighbor_count(corner)
    assert int(counts[0, 1]) == 1
    assert int(counts[5, 5]) == 1
    assert int(counts[3, 3]) == 0
