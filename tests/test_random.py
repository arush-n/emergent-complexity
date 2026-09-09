import jax
import numpy as np

from emergent.core.random import generate_random_grids, key_from_seed, random_grid


def test_seed_reproducibility_and_extremes() -> None:
    first = random_grid(key_from_seed(42), 32, 24, 0.2)
    second = random_grid(key_from_seed(42), 32, 24, 0.2)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, random_grid(key_from_seed(43), 32, 24, 0.2))
    assert int(random_grid(key_from_seed(1), 4, 5, 0).sum()) == 0
    assert int(random_grid(key_from_seed(1), 4, 5, 1).sum()) == 20


def test_batched_random_grids_have_expected_shape() -> None:
    keys = jax.random.split(key_from_seed(11), 7)
    grids = generate_random_grids(keys, 9, 10, 0.5)
    assert grids.shape == (7, 9, 10)
