from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.rna_chemistry.alphabet import (
    BASES,
    hamming_distance,
)
from emergent.experiments.morphology_interactions.rna_chemistry.sequence import (
    decode_exact_shape_sequence,
    exact_shape_sequence,
    sequence_from_shape,
)


def test_local_surface_sequence_follows_exact_morphology_identity() -> None:
    shape = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    translated = np.asarray([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
    rotated = np.rot90(shape)

    first = sequence_from_shape(shape)
    second = sequence_from_shape(translated)
    third = sequence_from_shape(rotated)

    np.testing.assert_array_equal(first.bases, second.bases)
    np.testing.assert_array_equal(first.bases, third.bases)
    assert first.shape_key == second.shape_key == third.shape_key
    assert np.all((first.bases >= 0) & (first.bases < len(BASES)))


def test_local_surface_length_grows_with_reactive_boundary() -> None:
    small = sequence_from_shape(np.ones((1, 1), dtype=np.uint8))
    large = sequence_from_shape(np.ones((10, 10), dtype=np.uint8))

    assert small.length == 1
    assert large.length == 36
    assert large.length > small.length
    assert large.site_coordinates.shape == (large.length, 2)


def test_exact_shape_sequence_is_reversible() -> None:
    shape = np.asarray([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    sequence = exact_shape_sequence(sequence_from_shape(shape).shape_key)

    assert sequence.mode == "exact_shape"
    assert decode_exact_shape_sequence(sequence) == sequence.shape_key


def test_hamming_distance_is_quaternary_and_deterministic() -> None:
    assert hamming_distance("ACGU", "ACGA") == 1
    assert hamming_distance(np.asarray([0, 1]), np.asarray([0, 1])) == 0
