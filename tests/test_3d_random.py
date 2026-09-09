import jax
import jax.numpy as jnp

from emergent.core3d.random import (
    generate_random_grids_3d,
    random_grid_3d,
    random_rule_3d,
    random_rules_3d,
)
from emergent.core3d.rules import rule_to_int_3d


def test_random_3d_grid_seed_and_density_edges() -> None:
    key = jax.random.key(42)
    first = random_grid_3d(key, 4, 5, 6, 0.25)
    second = random_grid_3d(jax.random.key(42), 4, 5, 6, 0.25)
    assert first.dtype == jnp.uint8
    assert jnp.array_equal(first, second)
    assert int(random_grid_3d(key, 4, 5, 6, 0).sum()) == 0
    assert int(random_grid_3d(key, 4, 5, 6, 1).sum()) == 4 * 5 * 6


def test_random_3d_batch_and_rules_are_reproducible() -> None:
    keys = jax.random.split(jax.random.key(8), 3)
    grids = generate_random_grids_3d(keys, 3, 4, 5, 0.5)
    assert grids.shape == (3, 3, 4, 5)
    rules_a = random_rules_3d(jax.random.key(9), 4)
    rules_b = random_rules_3d(jax.random.key(9), 4)
    assert [rule_to_int_3d(rule) for rule in rules_a] == [rule_to_int_3d(rule) for rule in rules_b]
    assert len(random_rules_3d(jax.random.key(10), 0)) == 0
    assert 0 <= rule_to_int_3d(random_rule_3d(jax.random.key(11))) < 2**54
