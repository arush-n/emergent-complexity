"""Batched execution for many independent morphology-interaction universes."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import repeat
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...core.random import generate_random_grids, key_from_seed
from ...core.step import batched_step
from .canonical import ShapeKey
from .components import detect_components_batch
from .config import MorphologyExperimentConfig
from .engine import (
    InteractionStepContext,
    MorphologyInteractionEngine,
    initial_grid_from_config,
)
from .interaction import InteractionUniverse, make_interaction_universe
from .local_step import step_batch_with_interactions


@dataclass(frozen=True)
class ParallelExecutionConfig:
    """Resource controls for one lockstep batch run.

    ``pair_capacity`` is deliberately explicit.  Pair-rule tables are padded
    to this fixed size so JAX compiles once per batch shape.  The runner raises
    instead of silently dropping rules when a world exceeds the capacity.
    ``host_workers=0`` keeps host preparation serial, which is generally best
    for the Python-heavy canonicalization path.  Explicit values greater than
    one opt into host threads.  The JAX transition itself remains one batched
    call in the caller thread.
    """

    num_envs: int
    batch_size: int | None = None
    host_workers: int = 0
    pair_capacity: int = 256
    collect_metrics: bool = True
    record_snapshots: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if self.batch_size is not None and (
            not isinstance(self.batch_size, int)
            or self.batch_size < 1
            or self.batch_size > self.num_envs
        ):
            raise ValueError("batch_size must be between 1 and num_envs")
        if not isinstance(self.host_workers, int) or self.host_workers < 0:
            raise ValueError("host_workers must be a non-negative integer")
        if not isinstance(self.pair_capacity, int) or self.pair_capacity < 1:
            raise ValueError("pair_capacity must be a positive integer")
        for name in ("collect_metrics", "record_snapshots"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    @property
    def effective_batch_size(self) -> int:
        """Return the configured batch size, defaulting to all environments."""

        return self.num_envs if self.batch_size is None else self.batch_size

    @property
    def effective_host_workers(self) -> int:
        """Return the configured host worker count."""

        return max(1, self.host_workers)


@dataclass
class ParallelExperimentResult:
    """Final states and per-environment state retained by a parallel run."""

    config: MorphologyExperimentConfig
    seeds: tuple[int, ...]
    final_grids: np.ndarray
    engines: tuple[MorphologyInteractionEngine, ...]
    elapsed_seconds: float

    @property
    def environments_per_second(self) -> float:
        """Return completed environments per wall-clock second."""

        if self.elapsed_seconds <= 0.0:
            return float("inf")
        return len(self.engines) / self.elapsed_seconds

    @property
    def transitions_per_second(self) -> float:
        """Return completed environment-generations per wall-clock second."""

        if self.elapsed_seconds <= 0.0 or self.config.steps == 0:
            return float("inf")
        return len(self.engines) * self.config.steps / self.elapsed_seconds


def _resolve_seeds(
    config: MorphologyExperimentConfig,
    seeds: Any,
    num_envs: int,
) -> tuple[int, ...]:
    if seeds is None:
        return tuple(config.seed + index for index in range(num_envs))
    values = tuple(int(seed) for seed in seeds)
    if len(values) != num_envs:
        raise ValueError("seeds must contain exactly num_envs values")
    if any(seed < 0 for seed in values):
        raise ValueError("seeds must be non-negative")
    return values


def _initial_grid_batch(
    config: MorphologyExperimentConfig,
    seeds: tuple[int, ...],
    initial_grids: Any | None,
) -> np.ndarray:
    if initial_grids is not None:
        values = np.asarray(jax.device_get(initial_grids), dtype=np.uint8)
        expected = (len(seeds), config.height, config.width)
        if values.shape != expected:
            raise ValueError(f"initial_grids must have shape {expected}, got {values.shape}")
        return (values != 0).astype(np.uint8)

    if config.initial_condition == "api":
        raise ValueError("parallel API mode requires explicit initial_grids")
    if config.initial_condition == "random":
        keys = jnp.stack([key_from_seed(seed) for seed in seeds])
        grids = generate_random_grids(keys, config.height, config.width, config.density)
        return np.asarray(jax.device_get(grids), dtype=np.uint8)

    grids = [
        np.asarray(initial_grid_from_config(replace(config, seed=seed)), dtype=np.uint8)
        for seed in seeds
    ]
    return np.stack(grids, axis=0)


def _prepare_one(
    engine: MorphologyInteractionEngine,
    previous_grid: np.ndarray,
    pair_capacity: int,
    observation: tuple[list[Any], list[Any], int] | None = None,
) -> InteractionStepContext:
    detection_performed = _detection_is_due(engine)
    if detection_performed and observation is None:
        observation = engine._observe_current_grid(previous_grid)
    return engine._prepare_step(
        previous_grid,
        pair_capacity=pair_capacity,
        observation=observation,
    )


def _detection_is_due(engine: MorphologyInteractionEngine) -> bool:
    """Return whether this lockstep generation performs morphology analysis."""

    source_generation = engine.generation
    return (
        source_generation >= engine.config.warmup_steps
        and (source_generation - engine.config.warmup_steps) % engine.config.detect_every == 0
    )


def _batch_observations(
    engines: tuple[MorphologyInteractionEngine, ...],
    previous_grids: np.ndarray,
    canonical_cache: dict[tuple[tuple[int, int], bytes], ShapeKey],
) -> list[tuple[list[Any], list[Any], int]] | None:
    """Detect every environment once, then register each result independently."""

    if not engines or not _detection_is_due(engines[0]):
        return None
    components = detect_components_batch(
        previous_grids,
        min_component_cells=engines[0].config.min_component_cells,
        backend=engines[0].config.component_backend,
    )
    return [
        engine._observe_components(
            environment_components,
            canonical_cache=canonical_cache,
        )
        for engine, environment_components in zip(engines, components)
    ]


def _prepare_contexts(
    engines: tuple[MorphologyInteractionEngine, ...],
    previous_grids: np.ndarray,
    *,
    pair_capacity: int,
    executor: ThreadPoolExecutor | None,
    canonical_cache: dict[tuple[tuple[int, int], bytes], ShapeKey],
    analyze_morphology: bool,
) -> list[InteractionStepContext]:
    observations = (
        _batch_observations(engines, previous_grids, canonical_cache)
        if analyze_morphology
        else None
    )
    if observations is None:
        if analyze_morphology:
            observations = [None] * len(engines)
        else:
            # Supplying an empty observation intentionally bypasses host
            # detection for high-throughput, interaction-disabled controls.
            observations = [([], [], 0)] * len(engines)
    if executor is None:
        return [
            _prepare_one(engine, grid, pair_capacity, observation)
            for engine, grid, observation in zip(engines, previous_grids, observations)
        ]
    return list(
        executor.map(
            _prepare_one,
            engines,
            previous_grids,
            repeat(pair_capacity),
            observations,
        )
    )


def _pad_pair_tables(
    contexts: list[InteractionStepContext],
    pair_capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    birth_tables = np.zeros((len(contexts), pair_capacity, 9), dtype=np.uint8)
    survival_tables = np.zeros((len(contexts), pair_capacity, 9), dtype=np.uint8)
    for index, context in enumerate(contexts):
        if context.pair_birth_masks.shape[0] == 0:
            continue
        if context.pair_birth_masks.shape != (pair_capacity, 9):
            raise ValueError("prepared pair table does not match pair_capacity")
        birth_tables[index] = context.pair_birth_masks
        survival_tables[index] = context.pair_survival_masks
    return birth_tables, survival_tables


def run_parallel(
    config: MorphologyExperimentConfig,
    *,
    num_envs: int,
    seeds: Any | None = None,
    initial_grids: Any | None = None,
    shared_universe_seed: int | None = None,
    batch_size: int | None = None,
    host_workers: int = 0,
    pair_capacity: int = 256,
    collect_metrics: bool = True,
    record_snapshots: bool = False,
) -> ParallelExperimentResult:
    """Run many independent worlds with batched JAX transitions.

    Each environment retains its own morphology registry and interaction cache,
    while the current grids, owner maps, and pair-rule tables are stepped in a
    single batched JAX call per chunk.  Component labeling is batched on the
    host when the selected backend supports it; optional host threads process
    the remaining per-environment preparation in deterministic input order.
    Set ``shared_universe_seed`` to reuse one fixed interaction law across all
    environments; otherwise each environment uses its resolved seed as its
    universe seed, matching the scalar engine.
    """

    if not isinstance(config, MorphologyExperimentConfig):
        raise TypeError("config must be a MorphologyExperimentConfig")
    execution = ParallelExecutionConfig(
        num_envs=num_envs,
        batch_size=batch_size,
        host_workers=host_workers,
        pair_capacity=pair_capacity,
        collect_metrics=collect_metrics,
        record_snapshots=record_snapshots,
    )
    resolved_seeds = _resolve_seeds(config, seeds, execution.num_envs)
    initial_batch = _initial_grid_batch(config, resolved_seeds, initial_grids)

    shared_universe: InteractionUniverse | None = None
    if shared_universe_seed is not None:
        if not isinstance(shared_universe_seed, int) or shared_universe_seed < 0:
            raise ValueError("shared_universe_seed must be a non-negative integer")
        shared_universe = make_interaction_universe(
            shared_universe_seed,
            identity_dim=config.identity_dim,
            beta=config.beta,
        )

    engines = tuple(
        MorphologyInteractionEngine(
            replace(config, seed=seed),
            initial_grid=initial_batch[index],
            universe=shared_universe,
        )
        for index, seed in enumerate(resolved_seeds)
    )
    if not execution.record_snapshots:
        for engine in engines:
            engine.snapshots.clear()

    canonical_cache: dict[tuple[tuple[int, int], bytes], ShapeKey] = {}
    analyze_morphology = config.interactions_enabled or execution.collect_metrics
    worker_count = execution.effective_host_workers
    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    start_time = time.perf_counter()
    try:
        final_batch = initial_batch.copy()
        for chunk_start in range(0, execution.num_envs, execution.effective_batch_size):
            chunk_end = min(
                execution.num_envs,
                chunk_start + execution.effective_batch_size,
            )
            chunk_engines = engines[chunk_start:chunk_end]
            current = jnp.asarray(initial_batch[chunk_start:chunk_end], dtype=jnp.uint8)
            placeholder_host = np.zeros_like(initial_batch[chunk_start:chunk_end], dtype=np.uint8)
            for _ in range(config.steps):
                source_generation = chunk_engines[0].generation
                next_generation = source_generation + 1
                analysis_due = analyze_morphology and _detection_is_due(chunk_engines[0])
                metrics_due = (
                    execution.collect_metrics and next_generation % config.metrics_every == 0
                )
                previous_host = (
                    np.asarray(jax.device_get(current), dtype=np.uint8)
                    if analysis_due or metrics_due
                    else placeholder_host
                )
                contexts = _prepare_contexts(
                    chunk_engines,
                    previous_host,
                    pair_capacity=execution.pair_capacity,
                    executor=executor,
                    canonical_cache=canonical_cache,
                    analyze_morphology=analyze_morphology,
                )
                if any(context.pair_birth_masks.shape[0] for context in contexts):
                    pair_birth, pair_survival = _pad_pair_tables(
                        contexts,
                        execution.pair_capacity,
                    )
                    owner_maps = np.stack([context.owner_map for context in contexts], axis=0)
                    next_batch = step_batch_with_interactions(
                        current,
                        owner_maps,
                        pair_birth,
                        pair_survival,
                        chunk_engines[0].base_birth,
                        chunk_engines[0].base_survival,
                    )
                else:
                    next_batch = batched_step(
                        current,
                        chunk_engines[0].base_birth,
                        chunk_engines[0].base_survival,
                    )
                host_grid_needed = (metrics_due) or (
                    execution.record_snapshots and next_generation % config.snapshot_every == 0
                )
                next_host = (
                    np.asarray(jax.device_get(next_batch), dtype=np.uint8)
                    if host_grid_needed
                    else None
                )
                for index, (engine, context) in enumerate(zip(chunk_engines, contexts)):
                    engine._finish_step(
                        context,
                        next_batch[index],
                        next_host=None if next_host is None else next_host[index],
                        record_metrics=execution.collect_metrics,
                        record_snapshot=execution.record_snapshots,
                        record_step_result=False,
                    )
                current = next_batch
            final_batch[chunk_start:chunk_end] = np.asarray(
                jax.device_get(current),
                dtype=np.uint8,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    elapsed = time.perf_counter() - start_time
    return ParallelExperimentResult(
        config=config,
        seeds=resolved_seeds,
        final_grids=(final_batch != 0).astype(np.uint8),
        engines=engines,
        elapsed_seconds=elapsed,
    )


ParallelEngine = run_parallel


def build_parser() -> argparse.ArgumentParser:
    """Build the optional many-environment CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument(
        "--component-backend",
        choices=("auto", "python", "scipy"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--host-workers", type=int, default=0)
    parser.add_argument("--pair-capacity", type=int, default=256)
    parser.add_argument("--shared-universe-seed", type=int, default=None)
    parser.add_argument("--disable-interactions", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a compact parallel benchmark and print throughput."""

    args = build_parser().parse_args(argv)
    config = MorphologyExperimentConfig(
        width=args.size,
        height=args.size,
        seed=args.seed,
        density=args.density,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        alpha=args.alpha,
        detect_every=args.detect_every,
        metrics_every=args.metrics_every,
        component_backend=args.component_backend,
        interactions_enabled=not args.disable_interactions,
    )
    result = run_parallel(
        config,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        host_workers=args.host_workers,
        pair_capacity=args.pair_capacity,
        shared_universe_seed=args.shared_universe_seed,
        collect_metrics=not args.no_metrics,
    )
    print(
        {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "num_envs": args.num_envs,
            "steps": args.steps,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
            "environments_per_second": round(result.environments_per_second, 3),
            "transitions_per_second": round(result.transitions_per_second, 3),
            "final_alive_mean": round(float(result.final_grids.sum(axis=(1, 2)).mean()), 3),
        }
    )


if __name__ == "__main__":
    main()
