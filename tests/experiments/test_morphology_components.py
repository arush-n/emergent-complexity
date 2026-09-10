from __future__ import annotations

import numpy as np
import pytest

from emergent.experiments.morphology_interactions.components import (
    detect_components,
    detect_components_batch,
)
from emergent.experiments.morphology_interactions.interaction import (
    component_pair_is_close,
    find_interacting_pairs,
)


def test_components_use_toroidal_eight_connectivity() -> None:
    grid = np.zeros((6, 8), dtype=np.uint8)
    grid[1, 1] = 1
    grid[2, 2] = 1  # diagonal Moore neighbor
    grid[4, 0] = 1
    grid[4, 7] = 1  # horizontal wraparound neighbor

    components = detect_components(grid)

    assert [component.cell_count for component in components] == [2, 2]
    assert components[0].coordinates.tolist() == [[1, 1], [2, 2]]
    assert components[1].coordinates.tolist() == [[4, 0], [4, 7]]


def test_minimum_component_size_filters_without_changing_connectivity() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[0, 0] = 1
    grid[2, 2] = 1
    grid[2, 3] = 1

    components = detect_components(grid, min_component_cells=2)

    assert len(components) == 1
    assert components[0].cell_count == 2


def test_spatial_encounters_use_toroidal_chebyshev_distance() -> None:
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[3, 0] = 1
    grid[3, 6] = 1  # two cells apart through the toroidal boundary
    components = detect_components(grid)

    assert component_pair_is_close(components[0], components[1], grid.shape, 2)
    assert find_interacting_pairs(components, grid.shape, 2) == [(0, 1)]


def test_batched_scipy_detector_matches_reference_toroidal_detector() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(123)
    grids = (rng.random((7, 13, 11)) < 0.18).astype(np.uint8)
    grids[0, 0, 0] = 1
    grids[0, -1, -1] = 1
    grids[1, 0, -1] = 1
    grids[1, -1, 0] = 1

    expected = [detect_components(grid, min_component_cells=2, backend="python") for grid in grids]
    actual = detect_components_batch(
        grids,
        min_component_cells=2,
        backend="scipy",
    )

    for expected_components, actual_components in zip(expected, actual):
        assert [component.cell_count for component in actual_components] == [
            component.cell_count for component in expected_components
        ]
        assert [component.coordinates.tolist() for component in actual_components] == [
            component.coordinates.tolist() for component in expected_components
        ]


def test_vectorized_interacting_pairs_match_component_distance_reference() -> None:
    rng = np.random.default_rng(456)
    for height, width in ((1, 1), (2, 3), (5, 7), (9, 4)):
        for radius in (0, 1, 2, 4):
            for _ in range(12):
                grid = (rng.random((height, width)) < 0.28).astype(np.uint8)
                components = detect_components(grid, backend="python")
                expected = {
                    tuple(sorted((first, second)))
                    for first in range(len(components))
                    for second in range(first + 1, len(components))
                    if component_pair_is_close(
                        components[first], components[second], grid.shape, radius
                    )
                }
                assert find_interacting_pairs(components, grid.shape, radius) == sorted(expected)
