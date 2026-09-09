import jax.numpy as jnp

from emergent.core3d.rules import parse_rule_3d, rule_to_masks_3d
from emergent.core3d.simulate import (
    generate_batched_trajectory_3d,
    generate_trajectory_3d,
    run_batched_until_stable_3d,
    run_steps_3d,
    run_steps_batch_3d,
    run_steps_batch_with_metrics_3d,
    run_steps_with_metrics_3d,
)
from emergent.core3d.step import step_3d


def test_3d_trajectory_starts_at_initial_and_follows_step() -> None:
    rule = parse_rule_3d("B1/S")
    initial = jnp.zeros((5, 5, 5), dtype=jnp.uint8).at[2, 2, 2].set(1)
    trajectory = generate_trajectory_3d(initial, rule, steps=3)
    birth, survival = rule_to_masks_3d(rule)
    assert trajectory.shape == (4, 5, 5, 5)
    assert jnp.array_equal(trajectory[0], initial)
    for index in range(3):
        assert jnp.array_equal(trajectory[index + 1], step_3d(trajectory[index], birth, survival))


def test_record_every_keeps_initial_and_final_partial_frame() -> None:
    rule = parse_rule_3d("B1/S")
    initial = jnp.zeros((4, 4, 4), dtype=jnp.uint8).at[1, 1, 1].set(1)
    trajectory = generate_trajectory_3d(initial, rule, steps=5, record_every=2)
    assert trajectory.shape[0] == 4  # t=0, 2, 4, and final t=5
    assert jnp.array_equal(trajectory[-1], run_steps_3d(initial, rule, 5))


def test_batched_3d_trajectory_and_metrics_shapes() -> None:
    rule = parse_rule_3d("B1/S")
    initial = jnp.zeros((2, 4, 4, 4), dtype=jnp.uint8).at[0, 1, 1, 1].set(1)
    initial = initial.at[1, 2, 2, 2].set(1)
    trajectory = generate_batched_trajectory_3d(initial, rule, steps=2)
    assert trajectory.shape == (2, 3, 4, 4, 4)
    final, metrics = run_steps_batch_with_metrics_3d(initial, rule, steps=2)
    assert final.shape == (2, 4, 4, 4)
    assert metrics.shape == (2, 2, 4)
    assert jnp.array_equal(final, run_steps_batch_3d(initial, rule, steps=2))


def test_3d_empty_grid_reaches_fixed_point() -> None:
    rule = parse_rule_3d("B/S")
    initial = jnp.zeros((4, 4, 4), dtype=jnp.uint8)
    final, steps, settled = run_batched_until_stable_3d(initial[None, ...], rule, max_steps=20)
    assert jnp.array_equal(final[0], initial)
    assert int(steps[0]) == 1
    assert bool(settled[0])


def test_single_metrics_include_births_and_deaths() -> None:
    rule = parse_rule_3d("B1/S")
    initial = jnp.zeros((5, 5, 5), dtype=jnp.uint8).at[2, 2, 1].set(1)
    _, metrics = run_steps_with_metrics_3d(initial, rule, steps=1)
    assert metrics.shape == (1, 4)
    assert int(metrics[0, 2]) == 26
