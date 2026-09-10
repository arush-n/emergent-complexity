from __future__ import annotations

import numpy as np

from emergent.experiments.morphology_interactions.rna_chemistry.accessibility import (
    AccessibilityCache,
    compute_accessibility,
    fold_pairs,
)
from emergent.experiments.morphology_interactions.rna_chemistry.sequence import sequence_from_shape


def test_none_accessibility_exposes_every_position() -> None:
    values = compute_accessibility([0, 1, 2, 3], mode="none")
    np.testing.assert_array_equal(values, np.ones(4))


def test_simplified_fold_is_deterministic_and_bounded() -> None:
    values = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
    first = compute_accessibility(values, mode="simplified_fold")
    second = compute_accessibility(values, mode="simplified_fold")

    np.testing.assert_array_equal(first, second)
    assert np.all((0.25 <= first) & (first <= 1.0))
    pairs = fold_pairs(values)
    assert pairs == fold_pairs(values)
    assert len(pairs) <= values.size // 2
    assert len({index for pair in pairs for index in pair}) == 2 * len(pairs)


def test_accessibility_cache_is_keyed_by_shape_key() -> None:
    sequence = sequence_from_shape(np.asarray([[1, 1], [1, 0]], dtype=np.uint8))
    cache = AccessibilityCache(mode="simplified_fold")
    first = cache.get(sequence)
    second = cache.get(sequence)

    assert len(cache.profiles) == 1
    assert first is second


def test_long_sequences_use_the_bounded_jax_window_path() -> None:
    values = np.resize(np.asarray([0, 1, 2, 3], dtype=np.uint8), 300)
    result = compute_accessibility(values, mode="simplified_fold", max_nussinov_length=32)

    assert result.shape == (300,)
    assert np.all(np.isfinite(result))
