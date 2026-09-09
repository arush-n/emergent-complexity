import jax.numpy as jnp
import numpy as np
import pytest

from emergent.core.measurements import (
    alive_count,
    alive_fraction,
    changed_cell_count,
    has_changed,
    transition_counts,
)
from emergent.core.rules import parse_rule
from emergent.core.simulate import (
    generate_rule_trajectories,
    generate_trajectory,
    run_batched_until_stable,
    run_steps,
    run_steps_dynamic_with_metrics,
    run_steps_with_metrics,
    run_until_stable,
    simulate_rules,
)


def test_trajectory_contains_initial_and_successive_steps() -> None:
    rule = parse_rule("B3/S23")
    initial = jnp.zeros((8, 8), dtype=jnp.uint8).at[3, 3].set(1)
    trajectory = generate_trajectory(initial, rule, 4)
    assert trajectory.shape == (5, 8, 8)
    np.testing.assert_array_equal(trajectory[0], initial)
    for previous, current in zip(trajectory[:-1], trajectory[1:]):
        np.testing.assert_array_equal(current, run_steps(previous, rule, 1))


def test_zero_step_and_measurements() -> None:
    rule = parse_rule("B3/S23")
    initial = jnp.eye(5, dtype=jnp.uint8)
    np.testing.assert_array_equal(run_steps(initial, rule, 0), initial)
    assert int(alive_count(initial)) == 5
    assert float(alive_fraction(initial)) == pytest.approx(0.2)
    assert int(changed_cell_count(initial, jnp.zeros_like(initial))) == 5
    assert bool(has_changed(initial, jnp.zeros_like(initial)))
    assert not bool(has_changed(initial, initial))


def test_transition_counts_report_births_and_deaths() -> None:
    previous = jnp.array([[1, 0], [0, 0]], dtype=jnp.uint8)
    current = jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8)
    np.testing.assert_array_equal(transition_counts(previous, current), [2, 3, 2, 1])


def test_metrics_scan_matches_trajectory() -> None:
    rule = parse_rule("B3/S23")
    initial = jnp.zeros((5, 5), dtype=jnp.uint8).at[2, 1:4].set(1)
    final, metrics = run_steps_with_metrics(initial, rule, 2)
    trajectory = generate_trajectory(initial, rule, 2)
    np.testing.assert_array_equal(final, trajectory[-1])
    np.testing.assert_array_equal(metrics[:, 0], [3, 3])
    np.testing.assert_array_equal(metrics[:, 1:], [[4, 2, 2], [4, 2, 2]])


def test_dynamic_metrics_path_matches_fixed_scan() -> None:
    rule = parse_rule("B3/S23")
    initial = jnp.zeros((8, 8), dtype=jnp.uint8).at[3, 2:5].set(1)
    final, last_metrics = run_steps_dynamic_with_metrics(initial, rule, 7)
    trajectory = generate_trajectory(initial, rule, 7)
    np.testing.assert_array_equal(final, trajectory[-1])
    np.testing.assert_array_equal(last_metrics, transition_counts(trajectory[-2], trajectory[-1]))


def test_run_until_stable_stops_at_a_fixed_point() -> None:
    rule = parse_rule("B3/S23")
    block = jnp.zeros((5, 5), dtype=jnp.uint8).at[1:3, 1:3].set(1)
    final, steps, settled = run_until_stable(block, rule, max_steps=50)
    np.testing.assert_array_equal(final, block)
    assert int(steps) == 1
    assert bool(settled)


def test_run_batched_until_stable_stops_each_grid_independently() -> None:
    rule = parse_rule("B3/S23")
    block = jnp.zeros((5, 5), dtype=jnp.uint8).at[1:3, 1:3].set(1)
    single = jnp.zeros((5, 5), dtype=jnp.uint8).at[2, 2].set(1)
    finals, steps, settled = run_batched_until_stable(jnp.stack([block, single]), rule, 20)
    np.testing.assert_array_equal(finals[0], block)
    np.testing.assert_array_equal(finals[1], jnp.zeros((5, 5), dtype=jnp.uint8))
    np.testing.assert_array_equal(steps, [1, 2])
    np.testing.assert_array_equal(settled, [True, True])


def test_many_rules_return_rule_major_final_states() -> None:
    initial_grids = jnp.zeros((3, 8, 8), dtype=jnp.uint8).at[:, 3, 3].set(1)
    rules = [parse_rule("B3/S23"), parse_rule("B36/S23")]
    finals = simulate_rules(rules, initial_grids, 2)
    assert finals.shape == (2, 3, 8, 8)


def test_many_rule_trajectories_have_explicit_layout() -> None:
    initial_grids = jnp.zeros((2, 6, 6), dtype=jnp.uint8)
    trajectories = generate_rule_trajectories(
        [parse_rule("B3/S23"), parse_rule("B36/S23")],
        initial_grids,
        3,
    )
    assert trajectories.shape == (2, 2, 4, 6, 6)
    np.testing.assert_array_equal(
        trajectories[:, :, 0],
        jnp.broadcast_to(initial_grids, (2, 2, 6, 6)),
    )
