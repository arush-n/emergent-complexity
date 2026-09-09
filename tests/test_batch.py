import jax
import jax.numpy as jnp
import numpy as np

from emergent.core.random import generate_random_grids, key_from_seed
from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.simulate import generate_batched_trajectory, simulate_rule_batch
from emergent.core.step import batched_step, step


def test_batched_step_matches_individual_steps() -> None:
    keys = jax.random.split(key_from_seed(42), 6)
    grids = generate_random_grids(keys, 16, 16, 0.25)
    birth, survival = rule_to_masks(parse_rule("B36/S23"))
    expected = jnp.stack([step(grid, birth, survival) for grid in grids])
    np.testing.assert_array_equal(batched_step(grids, birth, survival), expected)


def test_batched_trajectory_layout_and_final_state() -> None:
    grids = generate_random_grids(jax.random.split(key_from_seed(5), 4), 10, 12, 0.2)
    rule = parse_rule("B3/S23")
    trajectories = generate_batched_trajectory(grids, rule, 5)
    assert trajectories.shape == (4, 6, 10, 12)
    np.testing.assert_array_equal(trajectories[:, 0], grids)
    np.testing.assert_array_equal(trajectories[:, -1], simulate_rule_batch(rule, grids, 5))
