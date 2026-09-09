"""JAX-compiled multi-step and trajectory simulation APIs."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .grid import Grid
from .measurements import transition_counts
from .rules import Rule, rule_to_masks
from .step import batched_step, step


def _normalise_steps(steps: int) -> int:
    try:
        value = operator.index(steps)
    except TypeError as exc:
        raise TypeError("steps must be an integer") from exc
    if value < 0:
        raise ValueError("steps must be non-negative")
    return value


def _run_single_impl(
    initial_grid: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    steps: int,
) -> Grid:
    def scan_step(grid: Grid, _: None) -> tuple[Grid, None]:
        return step(grid, birth_mask, survival_mask), None

    final_grid, _ = jax.lax.scan(scan_step, initial_grid, None, length=steps)
    return final_grid


def _run_batch_impl(
    initial_grids: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    steps: int,
) -> Grid:
    def scan_step(grids: Grid, _: None) -> tuple[Grid, None]:
        return batched_step(grids, birth_mask, survival_mask), None

    final_grids, _ = jax.lax.scan(scan_step, initial_grids, None, length=steps)
    return final_grids


def _run_single_with_metrics_impl(
    initial_grid: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    steps: int,
) -> tuple[Grid, Grid]:
    def scan_step(grid: Grid, _: None) -> tuple[Grid, Grid]:
        new_grid = step(grid, birth_mask, survival_mask)
        return new_grid, transition_counts(grid, new_grid)

    return jax.lax.scan(scan_step, initial_grid, None, length=steps)


def _run_single_dynamic_with_metrics_impl(
    initial_grid: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    steps: int,
) -> tuple[Grid, Grid]:
    def body(_: int, carry: tuple[Grid, Grid]) -> tuple[Grid, Grid]:
        grid, _ = carry
        new_grid = step(grid, birth_mask, survival_mask)
        return new_grid, transition_counts(grid, new_grid)

    return jax.lax.fori_loop(
        0,
        steps,
        body,
        (initial_grid, jnp.zeros((4,), dtype=jnp.int32)),
    )


def _run_until_stable_impl(
    initial_grid: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    max_steps: int,
) -> tuple[Grid, Grid, Grid]:
    def condition(carry: tuple[Grid, Grid, Grid]) -> Grid:
        grid, steps, settled = carry
        return (steps < max_steps) & ~settled

    def body(carry: tuple[Grid, Grid, Grid]) -> tuple[Grid, Grid, Grid]:
        grid, steps, _ = carry
        new_grid = step(grid, birth_mask, survival_mask)
        settled = jnp.all(new_grid == grid)
        return new_grid, steps + 1, settled

    return jax.lax.while_loop(
        condition,
        body,
        (initial_grid, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)),
    )


def _run_until_stable_with_metrics_impl(
    initial_grid: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    max_steps: int,
) -> tuple[Grid, Grid, Grid, Grid]:
    def condition(carry: tuple[Grid, Grid, Grid, Grid]) -> Grid:
        grid, steps, settled, _ = carry
        return (steps < max_steps) & ~settled

    def body(carry: tuple[Grid, Grid, Grid, Grid]) -> tuple[Grid, Grid, Grid, Grid]:
        grid, steps, _, _ = carry
        new_grid = step(grid, birth_mask, survival_mask)
        metrics = transition_counts(grid, new_grid)
        settled = jnp.all(new_grid == grid)
        return new_grid, steps + 1, settled, metrics

    return jax.lax.while_loop(
        condition,
        body,
        (
            initial_grid,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((4,), dtype=jnp.int32),
        ),
    )


def _run_batch_until_stable_impl(
    initial_grids: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    max_steps: int,
) -> tuple[Grid, Grid, Grid]:
    batch_size = initial_grids.shape[0]

    def condition(carry: tuple[Grid, Grid, Grid, Grid]) -> Grid:
        _, _, _, active = carry
        return jnp.any(active)

    def body(carry: tuple[Grid, Grid, Grid, Grid]) -> tuple[Grid, Grid, Grid, Grid]:
        grids, steps, settled, active = carry
        can_advance = active & (steps < max_steps)
        next_grids = batched_step(grids, birth_mask, survival_mask)
        unchanged = jnp.all(next_grids == grids, axis=(1, 2))
        updated_grids = jnp.where(can_advance[:, None, None], next_grids, grids)
        updated_steps = steps + can_advance.astype(jnp.int32)
        updated_settled = settled | (can_advance & unchanged)
        updated_active = can_advance & ~unchanged
        return updated_grids, updated_steps, updated_settled, updated_active

    return jax.lax.while_loop(
        condition,
        body,
        (
            initial_grids,
            jnp.zeros((batch_size,), dtype=jnp.int32),
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.ones((batch_size,), dtype=jnp.bool_),
        ),
    )[:3]


def _trajectory_single_impl(
    initial_grid: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    steps: int,
) -> Grid:
    def scan_step(grid: Grid, _: None) -> tuple[Grid, Grid]:
        new_grid = step(grid, birth_mask, survival_mask)
        return new_grid, new_grid

    _, future_states = jax.lax.scan(scan_step, initial_grid, None, length=steps)
    return jnp.concatenate((initial_grid[jnp.newaxis, ...], future_states), axis=0)


def _trajectory_batch_impl(
    initial_grids: Grid,
    birth_mask: Grid,
    survival_mask: Grid,
    steps: int,
) -> Grid:
    def scan_step(grids: Grid, _: None) -> tuple[Grid, Grid]:
        new_grids = batched_step(grids, birth_mask, survival_mask)
        return new_grids, new_grids

    _, future_states = jax.lax.scan(scan_step, initial_grids, None, length=steps)
    time_major = jnp.concatenate((initial_grids[jnp.newaxis, ...], future_states), axis=0)
    return jnp.moveaxis(time_major, 0, 1)


run_steps_jit = jax.jit(_run_single_impl, static_argnames=("steps",))
run_batch_jit = jax.jit(_run_batch_impl, static_argnames=("steps",))
run_steps_with_metrics_jit = jax.jit(_run_single_with_metrics_impl, static_argnames=("steps",))
run_steps_dynamic_with_metrics_jit = jax.jit(_run_single_dynamic_with_metrics_impl)
run_until_stable_jit = jax.jit(_run_until_stable_impl)
run_until_stable_with_metrics_jit = jax.jit(_run_until_stable_with_metrics_impl)
run_batch_until_stable_jit = jax.jit(_run_batch_until_stable_impl)
generate_trajectory_jit = jax.jit(_trajectory_single_impl, static_argnames=("steps",))
generate_batched_trajectory_jit = jax.jit(
    _trajectory_batch_impl,
    static_argnames=("steps",),
)

_run_many_jit = jax.jit(
    jax.vmap(_run_batch_impl, in_axes=(None, 0, 0, None)),
    static_argnames=("steps",),
)
_generate_many_trajectories_jit = jax.jit(
    jax.vmap(_trajectory_batch_impl, in_axes=(None, 0, 0, None)),
    static_argnames=("steps",),
)


def _masks(rule: Rule) -> tuple[Grid, Grid]:
    return rule_to_masks(rule)


def run_steps(initial_grid: Grid, rule: Rule, steps: int) -> Grid:
    """Run a single grid for ``steps`` generations and return the final grid."""

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 2:
        raise ValueError(f"run_steps expects a grid with shape (height, width), got {grid.shape}")
    birth, survival = _masks(rule)
    return run_steps_jit(grid, birth, survival, step_count)


def simulate(initial_grid: Grid, rule: Rule, steps: int) -> Grid:
    """Alias for :func:`run_steps` used by the compact public API."""

    return run_steps(initial_grid, rule, steps)


def run_steps_with_metrics(initial_grid: Grid, rule: Rule, steps: int) -> tuple[Grid, Grid]:
    """Run steps and return ``(final_grid, metrics)``.

    ``metrics`` has shape ``(steps, 4)``. Each row is
    ``[alive, changed, births, deaths]`` for that generation. The transition
    remains entirely inside JAX, making this useful for UI traces and experiments.
    """

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 2:
        raise ValueError(
            f"run_steps_with_metrics expects a grid with shape (height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_steps_with_metrics_jit(grid, birth, survival, step_count)


def run_steps_dynamic_with_metrics(initial_grid: Grid, rule: Rule, steps: int) -> tuple[Grid, Grid]:
    """Run a dynamic number of steps without recompiling for each step count.

    Returns ``(final_grid, last_metrics)`` where ``last_metrics`` is the
    ``[alive, changed, births, deaths]`` row for the final generation. This is
    the server playback path; use :func:`run_steps_with_metrics` when every
    intermediate row is needed.
    """

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 2:
        raise ValueError(
            "run_steps_dynamic_with_metrics expects a grid with shape "
            f"(height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_steps_dynamic_with_metrics_jit(grid, birth, survival, step_count)


def run_until_stable(
    initial_grid: Grid,
    rule: Rule,
    max_steps: int = 1_000,
) -> tuple[Grid, Grid, Grid]:
    """Run until a fixed point or ``max_steps`` is reached.

    Returns ``(final_grid, steps_taken, settled)``. ``settled`` is false when
    the bound was reached first. Periodic states such as a blinker are not
    fixed points and therefore run until the bound.
    """

    step_count = _normalise_steps(max_steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 2:
        raise ValueError(
            f"run_until_stable expects a grid with shape (height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_until_stable_jit(grid, birth, survival, step_count)


def run_batched_until_stable(
    initial_grids: Grid,
    rule: Rule,
    max_steps: int = 1_000,
) -> tuple[Grid, Grid, Grid]:
    """Run a batch until each grid reaches a fixed point or the bound.

    Returns ``(final_grids, steps_taken, settled)``. Each batch member has its
    own stopping time, while all members advance in parallel on the same JAX
    device. Periodic states are bounded rather than classified.
    """

    step_count = _normalise_steps(max_steps)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 3:
        raise ValueError(
            f"run_batched_until_stable expects shape (batch, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return run_batch_until_stable_jit(grids, birth, survival, step_count)


def run_until_stable_with_metrics(
    initial_grid: Grid,
    rule: Rule,
    max_steps: int = 1_000,
) -> tuple[Grid, Grid, Grid, Grid]:
    """Run to a fixed point and return the final transition metrics too."""

    step_count = _normalise_steps(max_steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 2:
        raise ValueError(
            "run_until_stable_with_metrics expects a grid with shape "
            f"(height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_until_stable_with_metrics_jit(grid, birth, survival, step_count)


def generate_trajectory(initial_grid: Grid, rule: Rule, steps: int) -> Grid:
    """Return a single-grid trajectory with shape ``(steps + 1, height, width)``."""

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    birth, survival = _masks(rule)
    if grid.ndim == 2:
        return generate_trajectory_jit(grid, birth, survival, step_count)
    if grid.ndim == 3:
        return generate_batched_trajectory_jit(grid, birth, survival, step_count)
    raise ValueError(
        f"generate_trajectory expects a 2-D or batched 3-D grid, got shape {grid.shape}"
    )


def generate_batched_trajectory(initial_grids: Grid, rule: Rule, steps: int) -> Grid:
    """Return ``(batch, time, height, width)`` trajectories for many grids."""

    step_count = _normalise_steps(steps)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 3:
        raise ValueError(
            f"generate_batched_trajectory expects shape (batch, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return generate_batched_trajectory_jit(grids, birth, survival, step_count)


def simulate_rule(rule: Rule, initial_grid: Grid, steps: int) -> Grid:
    """Experiment-facing wrapper returning one rule's final single-grid state."""

    return run_steps(initial_grid, rule, steps)


def simulate_rule_batch(rule: Rule, initial_grids: Grid, steps: int) -> Grid:
    """Experiment-facing wrapper returning final states for one rule and a batch."""

    step_count = _normalise_steps(steps)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 3:
        raise ValueError(
            f"simulate_rule_batch expects shape (batch, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return run_batch_jit(grids, birth, survival, step_count)


def simulate_rules(rules: Sequence[Rule], initial_grids: Grid, steps: int) -> Grid:
    """Simulate many rules over the same initial batch.

    The returned array has shape ``(rule, batch, height, width)``. Rule parsing
    stays outside JIT; only numeric masks enter the compiled computation.
    """

    step_count = _normalise_steps(steps)
    rule_list = list(rules)
    if not rule_list:
        raise ValueError("rules must contain at least one rule")
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 3:
        raise ValueError(
            "simulate_rules expects initial_grids with shape (batch, height, width), "
            f"got {grids.shape}"
        )
    masks = [_masks(rule) for rule in rule_list]
    birth_masks = jnp.stack([birth for birth, _ in masks], axis=0)
    survival_masks = jnp.stack([survival for _, survival in masks], axis=0)
    return _run_many_jit(grids, birth_masks, survival_masks, step_count)


def generate_rule_trajectories(
    rules: Sequence[Rule],
    initial_grids: Grid,
    steps: int,
) -> Grid:
    """Generate ``(rule, batch, time, height, width)`` trajectories."""

    step_count = _normalise_steps(steps)
    rule_list = list(rules)
    if not rule_list:
        raise ValueError("rules must contain at least one rule")
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 3:
        raise ValueError(
            "generate_rule_trajectories expects initial_grids with shape "
            f"(batch, height, width), got {grids.shape}"
        )
    masks = [_masks(rule) for rule in rule_list]
    birth_masks = jnp.stack([birth for birth, _ in masks], axis=0)
    survival_masks = jnp.stack([survival for _, survival in masks], axis=0)
    return _generate_many_trajectories_jit(grids, birth_masks, survival_masks, step_count)
