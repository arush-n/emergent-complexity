from __future__ import annotations

import numpy as np

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import step_jit
from emergent.experiments.morphology_interactions.local_step import step_with_interactions


def test_no_interaction_zones_are_exactly_native_conway() -> None:
    grid = (np.arange(63, dtype=np.uint8).reshape(7, 9) % 2).astype(np.uint8)
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    owner_map = np.full(grid.shape, -1, dtype=np.int32)
    empty_birth = np.zeros((0, 9), dtype=np.uint8)
    empty_survival = np.zeros((0, 9), dtype=np.uint8)

    native = np.asarray(step_jit(grid, birth, survival))
    experimental = np.asarray(
        step_with_interactions(
            grid,
            owner_map,
            empty_birth,
            empty_survival,
            birth,
            survival,
        )
    )

    np.testing.assert_array_equal(experimental, native)
    assert experimental.dtype == np.uint8


def test_active_owner_map_uses_one_pair_rule_table_and_one_neighbor_count() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    owner_map = np.zeros(grid.shape, dtype=np.int32)
    local_birth, local_survival = rule_to_masks(parse_rule("B0/S"))
    base_birth, base_survival = rule_to_masks(parse_rule("B3/S23"))

    expected = np.asarray(step_jit(grid, local_birth, local_survival))
    actual = np.asarray(
        step_with_interactions(
            grid,
            owner_map,
            np.asarray(local_birth)[None, :],
            np.asarray(local_survival)[None, :],
            base_birth,
            base_survival,
        )
    )

    np.testing.assert_array_equal(actual, expected)
