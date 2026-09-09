from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.canonical import (
    canonical_matrix,
    canonical_matrix_from_grid,
    canonicalize_component,
    matrix_from_shape_key,
)
from emergent.experiments.morphology_interactions.components import Component


def test_translated_binary_shapes_have_the_same_exact_key() -> None:
    first = np.asarray(
        [
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    second = np.zeros((10, 12), dtype=np.uint8)
    second[6:8, 4:7] = first

    first_key = canonicalize_component(Component(np.argwhere(first), int(first.sum())))
    second_key = canonicalize_component(
        Component(np.argwhere(second), int(second.sum())),
    )

    assert first_key == second_key
    np.testing.assert_array_equal(
        matrix_from_shape_key(first_key),
        canonical_matrix_from_grid(first),
    )


def test_rotations_match_only_when_rotation_invariance_is_enabled() -> None:
    shape = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    rotated = np.rot90(shape)

    assert (
        canonical_matrix_from_grid(
            shape,
            rotation_invariant=True,
        ).tolist()
        == canonical_matrix_from_grid(rotated, rotation_invariant=True).tolist()
    )
    assert (
        canonical_matrix_from_grid(
            shape,
            rotation_invariant=False,
        ).tolist()
        != canonical_matrix_from_grid(rotated, rotation_invariant=False).tolist()
    )


def test_reflections_remain_distinct_by_default_but_can_be_merged() -> None:
    shape = np.asarray([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    reflected = np.fliplr(shape)

    assert canonicalize_component(np.argwhere(shape)) != canonicalize_component(
        np.argwhere(reflected)
    )
    assert canonicalize_component(
        np.argwhere(shape), reflection_invariant=True
    ) == canonicalize_component(np.argwhere(reflected), reflection_invariant=True)


def test_toroidal_unwrap_recovers_a_translated_wrapping_shape() -> None:
    wrapped = np.asarray([[3, 0], [3, 11], [4, 0]], dtype=np.int64)
    translated = np.asarray([[6, 3], [6, 4], [7, 4]], dtype=np.int64)

    wrapped_key = canonicalize_component(
        Component(wrapped, 3),
        grid_shape=(12, 12),
        rotation_invariant=False,
    )
    translated_key = canonicalize_component(
        Component(translated, 3),
        grid_shape=(12, 12),
        rotation_invariant=False,
    )

    assert wrapped_key == translated_key
    assert canonical_matrix(wrapped, grid_shape=(12, 12), rotation_invariant=False).tolist() == [
        [1, 1],
        [0, 1],
    ]
