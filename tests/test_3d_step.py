import jax
import jax.numpy as jnp

from emergent.core3d.rules import parse_rule_3d, rule_to_masks_3d
from emergent.core3d.step import batched_step_3d, count_neighbors_3d, step_3d, step_3d_jit


def test_full_3d_neighborhood_has_26_neighbors() -> None:
    grid = jnp.ones((3, 3, 3), dtype=jnp.uint8)
    assert int(count_neighbors_3d(grid)[1, 1, 1]) == 26


def test_3d_neighbor_count_handles_zero_one_six_and_wraparound() -> None:
    grid = jnp.zeros((5, 5, 5), dtype=jnp.uint8)
    grid = grid.at[2, 2, 1].set(1)
    grid = grid.at[2, 1, 2].set(1)
    grid = grid.at[1, 2, 2].set(1)
    grid = grid.at[2, 2, 3].set(1)
    grid = grid.at[2, 3, 2].set(1)
    grid = grid.at[3, 2, 2].set(1)
    counts = count_neighbors_3d(grid)
    assert int(counts[2, 2, 2]) == 6
    assert int(counts[0, 0, 0]) == 0

    wrapped = jnp.zeros((5, 5, 5), dtype=jnp.uint8).at[0, 0, 0].set(1)
    assert int(count_neighbors_3d(wrapped)[4, 4, 4]) == 1


def test_3d_birth_and_survival() -> None:
    birth_rule = parse_rule_3d("B1/S")
    birth, survival = rule_to_masks_3d(birth_rule)
    one_neighbor = jnp.zeros((5, 5, 5), dtype=jnp.uint8).at[2, 2, 1].set(1)
    born = step_3d(one_neighbor, birth, survival)
    assert int(born[2, 2, 2]) == 1

    survival_rule = parse_rule_3d("B/S1")
    birth, survival = rule_to_masks_3d(survival_rule)
    living = one_neighbor.at[2, 2, 2].set(1)
    survives = step_3d(living, birth, survival)
    assert int(survives[2, 2, 2]) == 1


def test_3d_jit_and_batch_match_plain_step() -> None:
    rule = parse_rule_3d("B1,3/S2,4")
    birth, survival = rule_to_masks_3d(rule)
    grid = (jax.random.uniform(jax.random.key(7), (5, 5, 5)) > 0.8).astype(jnp.uint8)
    plain = step_3d(grid, birth, survival)
    assert jnp.array_equal(plain, step_3d_jit(grid, birth, survival))
    batch = jnp.stack((grid, jnp.roll(grid, 1, axis=0)))
    expected = jnp.stack(tuple(step_3d(item, birth, survival) for item in batch))
    assert jnp.array_equal(expected, batched_step_3d(batch, birth, survival))
