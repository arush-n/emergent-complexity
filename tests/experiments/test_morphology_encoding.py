"""Tests for exact and size-aware morphology representations."""

import numpy as np
import pytest

from emergent.experiments.morphology_interactions.canonical import canonicalize_grid
from emergent.experiments.morphology_interactions.encoding import (
    audit_encodings,
    exact_identity_vector,
    make_encoder,
)


def test_exact_identity_vector_is_translation_invariant_and_injective() -> None:
    first = np.asarray(
        [
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    translated = np.pad(first, ((3, 1), (2, 4)))
    different = np.asarray(
        [
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )

    first_key = canonicalize_grid(first)
    translated_key = canonicalize_grid(translated)
    different_key = canonicalize_grid(different)

    first_vector = exact_identity_vector(first_key)
    translated_vector = exact_identity_vector(translated_key)
    different_vector = exact_identity_vector(different_key)

    assert first_key == translated_key
    assert np.array_equal(first_vector, translated_vector)
    assert first_key != different_key
    assert not np.array_equal(first_vector, different_vector)
    assert first_vector.size == 2 + first_key.height * first_key.width


def test_scaled_encoding_norm_tracks_cell_count() -> None:
    encoder = make_encoder(universe_seed=42, identity_dim=32)
    shapes = [np.ones((side, side), dtype=np.uint8) for side in (1, 2, 3, 4)]

    norms = [np.linalg.norm(encoder.encode_scaled(shape)) for shape in shapes]
    expected = [float(side) for side in (1, 2, 3, 4)]

    assert norms == pytest.approx(expected, abs=1e-6)


def test_scaled_encoding_rejects_invalid_exponents() -> None:
    encoder = make_encoder(universe_seed=42, identity_dim=8)
    shape = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="exponent"):
        encoder.encode_scaled(shape, exponent=-1.0)
    with pytest.raises(ValueError, match="exponent"):
        encoder.encode_scaled(shape, exponent=np.inf)


def test_encoding_audit_reports_exact_and_empirical_counts() -> None:
    keys = [
        canonicalize_grid(np.asarray([[1]], dtype=np.uint8)),
        canonicalize_grid(np.asarray([[1, 1]], dtype=np.uint8)),
    ]

    audit = audit_encodings(keys, universe_seed=42, identity_dim=8)

    assert audit["unique_shape_keys"] == 2
    assert audit["unique_exact_identity_vectors"] == 2
    assert audit["exact_identity_collisions"] == 0
    assert audit["unique_scaled_interaction_vectors"] == 2
    assert audit["scaled_vector_collisions"] == 0
    assert audit["cell_count_min"] == 1
    assert audit["cell_count_max"] == 2
