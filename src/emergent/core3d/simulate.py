"""JAX-compiled multi-step, batch, and trajectory APIs for 3D grids."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .grid import Grid3D
from .measurements import transition_counts_3d
from .rules import Rule3D, rule_to_masks_3d
from .step import batched_step_3d, step_3d


def _normalise_steps(steps: int) -> int:
    try:
        value = operator.index(steps)
    except TypeError as exc:
        raise TypeError("steps must be an integer") from exc
    if value < 0:
        raise ValueError("steps must be non-negative")
    return value


def _normalise_record_every(record_every: int) -> int:
    try:
        value = operator.index(record_every)
    except TypeError as exc:
        raise TypeError("record_every must be an integer") from exc
    if value < 1:
        raise ValueError("record_every must be positive")
    return value


def _run_single_impl(grid: Grid3D, birth: Grid3D, survival: Grid3D, steps: int) -> Grid3D:
    def scan_step(current: Grid3D, _: None) -> tuple[Grid3D, None]:
        return step_3d(current, birth, survival), None

    final, _ = jax.lax.scan(scan_step, grid, None, length=steps)
    return final


def _run_batch_impl(grids: Grid3D, birth: Grid3D, survival: Grid3D, steps: int) -> Grid3D:
    def scan_step(current: Grid3D, _: None) -> tuple[Grid3D, None]:
        return batched_step_3d(current, birth, survival), None

    final, _ = jax.lax.scan(scan_step, grids, None, length=steps)
    return final


def _run_single_with_metrics_impl(
    grid: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    steps: int,
) -> tuple[Grid3D, Grid3D]:
    def scan_step(current: Grid3D, _: None) -> tuple[Grid3D, Grid3D]:
        next_grid = step_3d(current, birth, survival)
        return next_grid, transition_counts_3d(current, next_grid)

    return jax.lax.scan(scan_step, grid, None, length=steps)


def _run_batch_with_metrics_impl(
    grids: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    steps: int,
) -> tuple[Grid3D, Grid3D]:
    def scan_step(current: Grid3D, _: None) -> tuple[Grid3D, Grid3D]:
        next_grids = batched_step_3d(current, birth, survival)
        metrics = jax.vmap(transition_counts_3d)(current, next_grids)
        return next_grids, metrics

    final, time_major_metrics = jax.lax.scan(scan_step, grids, None, length=steps)
    return final, jnp.moveaxis(time_major_metrics, 0, 1)


def _run_single_dynamic_with_metrics_impl(
    grid: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    steps: int,
) -> tuple[Grid3D, Grid3D]:
    def body(_: int, carry: tuple[Grid3D, Grid3D]) -> tuple[Grid3D, Grid3D]:
        current, _ = carry
        next_grid = step_3d(current, birth, survival)
        return next_grid, transition_counts_3d(current, next_grid)

    return jax.lax.fori_loop(
        0,
        steps,
        body,
        (grid, jnp.zeros((4,), dtype=jnp.int32)),
    )


def _trajectory_single_impl(
    grid: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    steps: int,
    record_every: int,
) -> Grid3D:
    full_chunks, remainder = divmod(steps, record_every)

    def advance_chunk(current: Grid3D, _: None) -> tuple[Grid3D, Grid3D]:
        def advance(_: int, value: Grid3D) -> Grid3D:
            return step_3d(value, birth, survival)

        final = jax.lax.fori_loop(0, record_every, advance, current)
        return final, final

    _, frames = jax.lax.scan(advance_chunk, grid, None, length=full_chunks)
    recorded = jnp.concatenate((grid[jnp.newaxis, ...], frames), axis=0)
    if remainder:

        def advance_remainder(_: int, value: Grid3D) -> Grid3D:
            return step_3d(value, birth, survival)

        final = jax.lax.fori_loop(0, remainder, advance_remainder, recorded[-1])
        recorded = jnp.concatenate((recorded, final[jnp.newaxis, ...]), axis=0)
    return recorded


def _trajectory_batch_impl(
    grids: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    steps: int,
    record_every: int,
) -> Grid3D:
    full_chunks, remainder = divmod(steps, record_every)

    def advance_chunk(current: Grid3D, _: None) -> tuple[Grid3D, Grid3D]:
        def advance(_: int, value: Grid3D) -> Grid3D:
            return batched_step_3d(value, birth, survival)

        final = jax.lax.fori_loop(0, record_every, advance, current)
        return final, final

    _, frames = jax.lax.scan(advance_chunk, grids, None, length=full_chunks)
    time_major = jnp.concatenate((grids[jnp.newaxis, ...], frames), axis=0)
    if remainder:

        def advance_remainder(_: int, value: Grid3D) -> Grid3D:
            return batched_step_3d(value, birth, survival)

        final = jax.lax.fori_loop(0, remainder, advance_remainder, time_major[-1])
        time_major = jnp.concatenate((time_major, final[jnp.newaxis, ...]), axis=0)
    return jnp.moveaxis(time_major, 0, 1)


def _run_until_stable_impl(
    grid: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    max_steps: int,
) -> tuple[Grid3D, Grid3D, Grid3D]:
    def condition(carry: tuple[Grid3D, Grid3D, Grid3D]) -> Grid3D:
        _, steps, settled = carry
        return (steps < max_steps) & ~settled

    def body(carry: tuple[Grid3D, Grid3D, Grid3D]) -> tuple[Grid3D, Grid3D, Grid3D]:
        current, steps, _ = carry
        next_grid = step_3d(current, birth, survival)
        return next_grid, steps + 1, jnp.all(next_grid == current)

    return jax.lax.while_loop(
        condition,
        body,
        (grid, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)),
    )


def _run_until_stable_with_metrics_impl(
    grid: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    max_steps: int,
) -> tuple[Grid3D, Grid3D, Grid3D, Grid3D]:
    def condition(carry: tuple[Grid3D, Grid3D, Grid3D, Grid3D]) -> Grid3D:
        _, steps, settled, _ = carry
        return (steps < max_steps) & ~settled

    def body(carry: tuple[Grid3D, Grid3D, Grid3D, Grid3D]) -> tuple[Grid3D, Grid3D, Grid3D, Grid3D]:
        current, steps, _, _ = carry
        next_grid = step_3d(current, birth, survival)
        metrics = transition_counts_3d(current, next_grid)
        return next_grid, steps + 1, jnp.all(next_grid == current), metrics

    return jax.lax.while_loop(
        condition,
        body,
        (
            grid,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((4,), dtype=jnp.int32),
        ),
    )


def _run_batch_until_stable_impl(
    grids: Grid3D,
    birth: Grid3D,
    survival: Grid3D,
    max_steps: int,
) -> tuple[Grid3D, Grid3D, Grid3D]:
    batch_size = grids.shape[0]

    def condition(carry: tuple[Grid3D, Grid3D, Grid3D, Grid3D]) -> Grid3D:
        _, _, _, active = carry
        return jnp.any(active)

    def body(carry: tuple[Grid3D, Grid3D, Grid3D, Grid3D]) -> tuple[Grid3D, Grid3D, Grid3D, Grid3D]:
        current, steps, settled, active = carry
        can_advance = active & (steps < max_steps)
        next_grids = batched_step_3d(current, birth, survival)
        unchanged = jnp.all(next_grids == current, axis=(1, 2, 3))
        updated_grids = jnp.where(can_advance[:, None, None, None], next_grids, current)
        updated_steps = steps + can_advance.astype(jnp.int32)
        updated_settled = settled | (can_advance & unchanged)
        updated_active = can_advance & ~unchanged
        return updated_grids, updated_steps, updated_settled, updated_active

    return jax.lax.while_loop(
        condition,
        body,
        (
            grids,
            jnp.zeros((batch_size,), dtype=jnp.int32),
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.ones((batch_size,), dtype=jnp.bool_),
        ),
    )[:3]


run_steps_3d_jit = jax.jit(_run_single_impl, static_argnames=("steps",))
run_steps_batch_3d_jit = jax.jit(_run_batch_impl, static_argnames=("steps",))
run_steps_with_metrics_3d_jit = jax.jit(_run_single_with_metrics_impl, static_argnames=("steps",))
run_steps_batch_with_metrics_3d_jit = jax.jit(
    _run_batch_with_metrics_impl,
    static_argnames=("steps",),
)
run_steps_dynamic_with_metrics_3d_jit = jax.jit(_run_single_dynamic_with_metrics_impl)
generate_trajectory_3d_jit = jax.jit(
    _trajectory_single_impl,
    static_argnames=("steps", "record_every"),
)
generate_batched_trajectory_3d_jit = jax.jit(
    _trajectory_batch_impl,
    static_argnames=("steps", "record_every"),
)
run_until_stable_3d_jit = jax.jit(_run_until_stable_impl)
run_until_stable_with_metrics_3d_jit = jax.jit(_run_until_stable_with_metrics_impl)
run_batched_until_stable_3d_jit = jax.jit(_run_batch_until_stable_impl)
_run_many_3d_jit = jax.jit(
    jax.vmap(_run_batch_impl, in_axes=(None, 0, 0, None)),
    static_argnames=("steps",),
)


def _masks(rule: Rule3D) -> tuple[Grid3D, Grid3D]:
    return rule_to_masks_3d(rule)


def run_steps_3d(initial_grid: Grid3D, rule: Rule3D, steps: int) -> Grid3D:
    """Run one 3D grid and return its final state."""

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError(f"run_steps_3d expects shape (depth, height, width), got {grid.shape}")
    birth, survival = _masks(rule)
    return run_steps_3d_jit(grid, birth, survival, step_count)


def run_steps_batch_3d(initial_grids: Grid3D, rule: Rule3D, steps: int) -> Grid3D:
    """Run a batch shaped ``(batch, depth, height, width)``."""

    step_count = _normalise_steps(steps)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 4:
        raise ValueError(
            f"run_steps_batch_3d expects shape (batch, depth, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return run_steps_batch_3d_jit(grids, birth, survival, step_count)


def run_steps_with_metrics_3d(
    initial_grid: Grid3D,
    rule: Rule3D,
    steps: int,
) -> tuple[Grid3D, Grid3D]:
    """Run a single grid and return ``(final_grid, metrics[time, 4])``."""

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError(
            f"run_steps_with_metrics_3d expects shape (depth, height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_steps_with_metrics_3d_jit(grid, birth, survival, step_count)


def run_steps_batch_with_metrics_3d(
    initial_grids: Grid3D,
    rule: Rule3D,
    steps: int,
) -> tuple[Grid3D, Grid3D]:
    """Run a batch and return metrics shaped ``(batch, time, 4)``."""

    step_count = _normalise_steps(steps)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 4:
        raise ValueError(
            "run_steps_batch_with_metrics_3d expects shape "
            f"(batch, depth, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return run_steps_batch_with_metrics_3d_jit(grids, birth, survival, step_count)


def run_steps_dynamic_with_metrics_3d(
    initial_grid: Grid3D,
    rule: Rule3D,
    steps: int,
) -> tuple[Grid3D, Grid3D]:
    """Run a dynamic step count without recompiling for each count."""

    step_count = _normalise_steps(steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError(
            "run_steps_dynamic_with_metrics_3d expects shape "
            f"(depth, height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_steps_dynamic_with_metrics_3d_jit(grid, birth, survival, step_count)


def generate_trajectory_3d(
    initial_grid: Grid3D,
    rule: Rule3D,
    steps: int,
    record_every: int = 1,
) -> Grid3D:
    """Return ``(time, depth, height, width)`` frames.

    The initial state is always included. With ``record_every > 1``, only every
    requested interval plus a final partial interval are retained.
    """

    step_count = _normalise_steps(steps)
    interval = _normalise_record_every(record_every)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError(
            f"generate_trajectory_3d expects shape (depth, height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return generate_trajectory_3d_jit(grid, birth, survival, step_count, interval)


def generate_batched_trajectory_3d(
    initial_grids: Grid3D,
    rule: Rule3D,
    steps: int,
    record_every: int = 1,
) -> Grid3D:
    """Return ``(batch, time, depth, height, width)`` frames."""

    step_count = _normalise_steps(steps)
    interval = _normalise_record_every(record_every)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 4:
        raise ValueError(
            "generate_batched_trajectory_3d expects shape "
            f"(batch, depth, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return generate_batched_trajectory_3d_jit(grids, birth, survival, step_count, interval)


def run_until_stable_3d(
    initial_grid: Grid3D,
    rule: Rule3D,
    max_steps: int = 1_000,
) -> tuple[Grid3D, Grid3D, Grid3D]:
    """Run until a fixed point or a safety bound is reached."""

    bound = _normalise_steps(max_steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError(
            f"run_until_stable_3d expects shape (depth, height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_until_stable_3d_jit(grid, birth, survival, bound)


def run_until_stable_with_metrics_3d(
    initial_grid: Grid3D,
    rule: Rule3D,
    max_steps: int = 1_000,
) -> tuple[Grid3D, Grid3D, Grid3D, Grid3D]:
    """Run to a fixed point and return the final transition metrics."""

    bound = _normalise_steps(max_steps)
    grid = jnp.asarray(initial_grid, dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError(
            "run_until_stable_with_metrics_3d expects shape "
            f"(depth, height, width), got {grid.shape}"
        )
    birth, survival = _masks(rule)
    return run_until_stable_with_metrics_3d_jit(grid, birth, survival, bound)


def run_batched_until_stable_3d(
    initial_grids: Grid3D,
    rule: Rule3D,
    max_steps: int = 1_000,
) -> tuple[Grid3D, Grid3D, Grid3D]:
    """Run a batch until each member reaches a fixed point or the bound."""

    bound = _normalise_steps(max_steps)
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 4:
        raise ValueError(
            "run_batched_until_stable_3d expects shape "
            f"(batch, depth, height, width), got {grids.shape}"
        )
    birth, survival = _masks(rule)
    return run_batched_until_stable_3d_jit(grids, birth, survival, bound)


def simulate_rule_3d(rule: Rule3D, initial_grid: Grid3D, steps: int) -> Grid3D:
    """Experiment-facing single-rule wrapper."""

    return run_steps_3d(initial_grid, rule, steps)


def simulate_rule_batch_3d(rule: Rule3D, initial_grids: Grid3D, steps: int) -> Grid3D:
    """Experiment-facing wrapper for one 3D rule over a batch."""

    return run_steps_batch_3d(initial_grids, rule, steps)


def simulate_rules_3d(rules: Sequence[Rule3D], initial_grids: Grid3D, steps: int) -> Grid3D:
    """Simulate many 3D rules over the same batch.

    The result has shape ``(rule, batch, depth, height, width)``. Rule parsing
    and Python dataclasses remain outside compiled JAX code.
    """

    step_count = _normalise_steps(steps)
    rule_list = list(rules)
    if not rule_list:
        raise ValueError("rules must contain at least one rule")
    grids = jnp.asarray(initial_grids, dtype=jnp.uint8)
    if grids.ndim != 4:
        raise ValueError(
            "simulate_rules_3d expects initial_grids with shape "
            f"(batch, depth, height, width), got {grids.shape}"
        )
    masks = [_masks(rule) for rule in rule_list]
    birth_masks = jnp.stack([birth for birth, _ in masks], axis=0)
    survival_masks = jnp.stack([survival for _, survival in masks], axis=0)
    return _run_many_3d_jit(grids, birth_masks, survival_masks, step_count)
