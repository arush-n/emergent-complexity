"""Standalone deterministic engine for morphology-dependent Life physics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...core.random import key_from_seed, random_grid
from ...core.rules import Rule, rule_to_masks
from ...core.step import step_jit
from ...io.patterns import get_pattern
from ...io.serialization import load_grid
from .canonical import ShapeKey, canonicalize_component
from .components import Component, detect_components
from .config import MorphologyExperimentConfig
from .interaction import (
    InteractionUniverse,
    PairInteraction,
    PairKey,
    build_interaction_zone,
    find_interacting_pairs,
    make_interaction_universe,
    make_pair_interaction,
    make_pair_key,
)
from .local_step import pair_rule_arrays, resolve_owner_map, step_with_interactions
from .metrics import MetricsTracker
from .native_api import load_native_grid
from .species import SpeciesRegistry


@dataclass(frozen=True)
class StepResult:
    """Host-side details for one deterministic ``X_t -> X_(t+1)`` transition."""

    source_generation: int
    generation: int
    grid: np.ndarray
    components: tuple[Component, ...]
    species_keys: tuple[ShapeKey, ...]
    active_interactions: tuple[PairInteraction, ...]
    owner_map: np.ndarray
    detection_performed: bool
    metrics: dict[str, int | float] | None


@dataclass(frozen=True)
class InteractionStepContext:
    """Prepared host-side fields for one deterministic transition."""

    source_generation: int
    previous_grid: np.ndarray
    components: tuple[Component, ...]
    species_keys: tuple[ShapeKey, ...]
    new_species_count: int
    active_interactions: tuple[PairInteraction, ...]
    owner_map: np.ndarray
    pair_birth_masks: np.ndarray
    pair_survival_masks: np.ndarray
    detection_performed: bool


@dataclass
class ExperimentResult:
    """Complete in-memory result handed to the persistence layer."""

    config: MorphologyExperimentConfig
    initial_grid: np.ndarray
    final_grid: np.ndarray
    universe: InteractionUniverse
    species_registry: SpeciesRegistry
    interaction_cache: dict[PairKey, PairInteraction]
    generation_records: list[dict[str, int | float]]
    snapshots: dict[int, np.ndarray]
    summary: dict[str, int | float]


def _host_grid(grid: Any) -> np.ndarray:
    values = np.asarray(jax.device_get(grid), dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError("initial grid must be two-dimensional")
    return (values != 0).astype(np.uint8)


def _place_pattern_toroidally(
    height: int,
    width: int,
    pattern_name: str,
    row: int | None,
    col: int | None,
) -> np.ndarray:
    pattern = np.asarray(get_pattern(pattern_name), dtype=np.uint8)
    result = np.zeros((height, width), dtype=np.uint8)
    start_row = (height - pattern.shape[0]) // 2 if row is None else row
    start_col = (width - pattern.shape[1]) // 2 if col is None else col
    for pattern_row, pattern_col in np.argwhere(pattern != 0):
        result[(start_row + int(pattern_row)) % height, (start_col + int(pattern_col)) % width] = 1
    return result


def initial_grid_from_config(config: MorphologyExperimentConfig) -> np.ndarray:
    """Create/load the deterministic starting grid selected by ``config``."""

    if config.initial_condition == "random":
        return _host_grid(
            random_grid(
                key_from_seed(config.seed),
                config.height,
                config.width,
                config.density,
            )
        )
    if config.initial_condition == "patterns":
        return _place_pattern_toroidally(
            config.height,
            config.width,
            config.pattern_name,
            config.pattern_row,
            config.pattern_col,
        )
    if config.initial_condition == "npz":
        return _host_grid(load_grid(config.initial_condition_path or ""))
    if config.initial_condition == "api":
        return _host_grid(
            load_native_grid(
                config.native_base_url,
                width=config.width,
                height=config.height,
                density=config.density,
                seed=config.seed,
                rule=config.base_rule,
            )
        )
    raise ValueError(f"unsupported initial_condition: {config.initial_condition}")


class MorphologyInteractionEngine:
    """Execute one isolated deterministic morphology-interaction universe."""

    def __init__(
        self,
        config: MorphologyExperimentConfig,
        *,
        initial_grid: Any | None = None,
        universe: InteractionUniverse | None = None,
    ) -> None:
        if not isinstance(config, MorphologyExperimentConfig):
            raise TypeError("config must be a MorphologyExperimentConfig")
        self.config = config
        self.base_rule: Rule = config.rule
        self.base_birth, self.base_survival = rule_to_masks(self.base_rule)
        if universe is None:
            self.universe = make_interaction_universe(
                config.seed,
                identity_dim=config.identity_dim,
                beta=config.beta,
            )
        else:
            if not isinstance(universe, InteractionUniverse):
                raise TypeError("universe must be an InteractionUniverse")
            if universe.identity_dim != config.identity_dim:
                raise ValueError("universe identity_dim must match config.identity_dim")
            if universe.beta != config.beta:
                raise ValueError("universe beta must match config.beta")
            self.universe = universe
        initial = (
            initial_grid_from_config(config) if initial_grid is None else _host_grid(initial_grid)
        )
        expected_shape = (config.height, config.width)
        if initial.shape != expected_shape:
            raise ValueError(f"initial grid shape must be {expected_shape}, got {initial.shape}")
        self.initial_grid = initial.copy()
        self.grid = jnp.asarray(initial, dtype=jnp.uint8)
        self.generation = 0
        self.species_registry = SpeciesRegistry()
        self.interaction_cache: dict[PairKey, PairInteraction] = {}
        self._canonical_cache: dict[tuple[tuple[int, int], bytes], ShapeKey] = {}
        # Keep the compiled pair-table shape stable across generations.  This
        # avoids one XLA compilation for every changing active-pair count while
        # retaining an explicit grow-on-demand fallback for unusual dense
        # worlds.
        self._pair_slot_capacity = max(1, config.width * config.height)
        self.metrics = MetricsTracker(config.height * config.width)
        self.snapshots: dict[int, np.ndarray] = {0: initial.copy()}
        self.last_step: StepResult | None = None

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def width(self) -> int:
        return self.config.width

    def _observe_current_grid(
        self,
        current: np.ndarray,
    ) -> tuple[list[Component], list[ShapeKey], int]:
        """Detect, canonicalize, register, and encode all current components."""

        components = detect_components(
            current,
            min_component_cells=self.config.min_component_cells,
            backend=self.config.component_backend,
        )
        return self._observe_components(components)

    def _observe_components(
        self,
        components: list[Component],
        *,
        canonical_cache: dict[tuple[tuple[int, int], bytes], ShapeKey] | None = None,
    ) -> tuple[list[Component], list[ShapeKey], int]:
        """Register a pre-detected component list without detecting twice."""

        cache = self._canonical_cache if canonical_cache is None else canonical_cache
        keys: list[ShapeKey] = []
        new_species_count = 0
        for component in components:
            coordinates = component.coordinates
            shifted = coordinates - coordinates.min(axis=0)
            cache_key: tuple[tuple[int, int], bytes] | None = None
            if (
                int(shifted[:, 0].max()) < self.height - 1
                and int(shifted[:, 1].max()) < self.width - 1
            ):
                cache_key = (tuple(shifted.shape), shifted.tobytes())
            key = cache.get(cache_key) if cache_key is not None else None
            if key is None:
                key = canonicalize_component(
                    component,
                    grid_shape=(self.height, self.width),
                    rotation_invariant=self.config.rotation_invariant,
                    reflection_invariant=self.config.reflection_invariant,
                )
                if cache_key is not None:
                    cache[cache_key] = key
            record, is_new = self.species_registry.observe(
                key,
                generation=self.generation,
                cell_count=component.cell_count,
            )
            self.species_registry.ensure_encoding(record, self.universe.encoder)
            keys.append(key)
            new_species_count += int(is_new)
        return components, keys, new_species_count

    def _cached_pair_interaction(
        self,
        key_a: ShapeKey,
        key_b: ShapeKey,
    ) -> PairInteraction:
        pair = make_pair_key(key_a, key_b)
        cached = self.interaction_cache.get(pair)
        if cached is None:
            record_a = self.species_registry[pair.species_a]
            record_b = self.species_registry[pair.species_b]
            vector_a = self.species_registry.ensure_encoding(record_a, self.universe.encoder)
            vector_b = self.species_registry.ensure_encoding(record_b, self.universe.encoder)
            cached = make_pair_interaction(
                pair.species_a,
                pair.species_b,
                vector_a,
                vector_b,
                universe=self.universe,
                alpha=self.config.alpha,
                base_rule=self.base_rule,
                max_rule_changes=self.config.max_rule_changes,
                interaction_threshold=self.config.interaction_threshold,
                first_seen_generation=self.generation,
            )
            self.interaction_cache[pair] = cached
        cached.encounters += 1
        return cached

    def _prepare_step(
        self,
        previous_grid: np.ndarray,
        *,
        pair_capacity: int | None = None,
        observation: tuple[list[Component], list[ShapeKey], int] | None = None,
    ) -> InteractionStepContext:
        """Prepare all deterministic host-side fields before a JAX step."""

        source_generation = self.generation
        detection_performed = (
            source_generation >= self.config.warmup_steps
            and (source_generation - self.config.warmup_steps) % self.config.detect_every == 0
        )
        components: list[Component] = []
        species_keys: list[ShapeKey] = []
        new_species_count = 0
        if detection_performed:
            if observation is None:
                components, species_keys, new_species_count = self._observe_current_grid(
                    previous_grid
                )
            else:
                components, species_keys, new_species_count = observation

        active_interactions: list[PairInteraction] = []
        unique_zones: dict[PairKey, np.ndarray] = {}
        interaction_is_live = (
            self.config.interactions_enabled
            and source_generation >= self.config.warmup_steps
            and detection_performed
        )
        if interaction_is_live:
            component_pairs = find_interacting_pairs(
                components,
                (self.height, self.width),
                self.config.interaction_radius,
            )
            for first_index, second_index in component_pairs:
                pair_interaction = self._cached_pair_interaction(
                    species_keys[first_index],
                    species_keys[second_index],
                )
                active_interactions.append(pair_interaction)
                pair = pair_interaction.pair_key
                zone = build_interaction_zone(
                    components[first_index],
                    components[second_index],
                    (self.height, self.width),
                    interaction_radius=self.config.interaction_radius,
                    effect_padding=self.config.effect_padding,
                )
                unique_zones[pair] = zone if pair not in unique_zones else unique_zones[pair] | zone

        if unique_zones:
            if pair_capacity is not None:
                if not isinstance(pair_capacity, int) or pair_capacity < 1:
                    raise ValueError("pair_capacity must be a positive integer")
                if len(unique_zones) > pair_capacity:
                    raise ValueError(
                        "pair_capacity is smaller than the active interaction count; "
                        "increase pair_capacity"
                    )
                table_capacity = pair_capacity
            else:
                if len(unique_zones) > self._pair_slot_capacity:
                    self._pair_slot_capacity = len(unique_zones)
                table_capacity = self._pair_slot_capacity
            strengths = {pair: self.interaction_cache[pair].strength for pair in unique_zones}
            owner_map, owner_pairs = resolve_owner_map(
                unique_zones,
                strengths=strengths,
                grid_shape=(self.height, self.width),
            )
            pair_birth_masks, pair_survival_masks = pair_rule_arrays(
                owner_pairs,
                self.interaction_cache,
                capacity=table_capacity,
            )
        else:
            owner_map = np.full((self.height, self.width), -1, dtype=np.int32)
            pair_birth_masks = np.zeros((0, 9), dtype=np.uint8)
            pair_survival_masks = np.zeros((0, 9), dtype=np.uint8)

        return InteractionStepContext(
            source_generation=source_generation,
            previous_grid=previous_grid.copy(),
            components=tuple(components),
            species_keys=tuple(species_keys),
            new_species_count=new_species_count,
            active_interactions=tuple(active_interactions),
            owner_map=owner_map,
            pair_birth_masks=pair_birth_masks,
            pair_survival_masks=pair_survival_masks,
            detection_performed=detection_performed,
        )

    def _finish_step(
        self,
        context: InteractionStepContext,
        next_grid: Any,
        *,
        next_host: np.ndarray | None = None,
        record_metrics: bool = True,
        record_snapshot: bool = True,
        record_step_result: bool = True,
    ) -> StepResult | None:
        """Commit one prepared transition and optionally retain diagnostics."""

        if record_metrics:
            self.metrics.observe_interactions(context.active_interactions)

        self.grid = next_grid
        self.generation += 1
        metrics_due = record_metrics and self.generation % self.config.metrics_every == 0
        snapshot_due = record_snapshot and self.generation % self.config.snapshot_every == 0
        needs_host_grid = record_step_result or metrics_due or snapshot_due
        if next_host is None and needs_host_grid:
            next_host = _host_grid(next_grid)
        elif next_host is not None:
            next_host = _host_grid(next_host)
        metrics_row: dict[str, int | float] | None = None
        if metrics_due:
            if next_host is None:  # pragma: no cover - guarded by needs_host_grid
                raise RuntimeError("a host grid is required for metrics")
            metrics_row = self.metrics.record(
                generation=self.generation,
                previous_grid=context.previous_grid,
                next_grid=next_host,
                components=context.components,
                species_keys=context.species_keys,
                new_species_count=context.new_species_count,
                registry=self.species_registry,
                active_interactions=context.active_interactions,
                interaction_cache=self.interaction_cache,
                detection_performed=context.detection_performed,
            )
        if snapshot_due:
            if next_host is None:  # pragma: no cover - guarded by needs_host_grid
                raise RuntimeError("a host grid is required for snapshots")
            self.snapshots[self.generation] = next_host.copy()

        if not record_step_result:
            self.last_step = None
            return None
        if next_host is None:  # pragma: no cover - guarded by needs_host_grid
            raise RuntimeError("a host grid is required for a step result")

        result = StepResult(
            source_generation=context.source_generation,
            generation=self.generation,
            grid=next_host.copy(),
            components=context.components,
            species_keys=context.species_keys,
            active_interactions=context.active_interactions,
            owner_map=context.owner_map.copy(),
            detection_performed=context.detection_performed,
            metrics=metrics_row,
        )
        self.last_step = result
        return result

    def step(self) -> StepResult:
        """Advance exactly one generation using the specified execution order."""

        previous_grid = _host_grid(self.grid)
        context = self._prepare_step(previous_grid)
        if context.pair_birth_masks.shape[0]:
            next_grid = step_with_interactions(
                self.grid,
                context.owner_map,
                context.pair_birth_masks,
                context.pair_survival_masks,
                self.base_birth,
                self.base_survival,
            )
        else:
            next_grid = step_jit(self.grid, self.base_birth, self.base_survival)
        return self._finish_step(context, next_grid)

    def run(
        self,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> ExperimentResult:
        """Run the configured number of transitions and return all compact state."""

        for _ in range(self.config.steps):
            self.step()
            if progress is not None:
                progress(self.generation, self.config.steps)
        final_host = _host_grid(self.grid)
        # A final frame is useful for validation and is still far smaller than
        # retaining a trajectory.  It does not add intermediate frames.
        self.snapshots.setdefault(self.generation, final_host.copy())
        return ExperimentResult(
            config=self.config,
            initial_grid=self.initial_grid.copy(),
            final_grid=final_host,
            universe=self.universe,
            species_registry=self.species_registry,
            interaction_cache=self.interaction_cache,
            generation_records=list(self.metrics.records),
            snapshots={generation: frame.copy() for generation, frame in self.snapshots.items()},
            summary=self.metrics.summary(
                final_host,
                self.species_registry,
                self.interaction_cache,
                last_generation=self.generation,
            ),
        )


Engine = MorphologyInteractionEngine


def run_experiment(
    config: MorphologyExperimentConfig,
    *,
    initial_grid: Any | None = None,
    output_dir: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExperimentResult:
    """Run one universe and optionally write its reproducible artifact bundle."""

    engine = MorphologyInteractionEngine(config, initial_grid=initial_grid)
    result = engine.run(progress=progress)
    from .persistence import default_output_directory, write_artifacts

    write_artifacts(result, output_dir or default_output_directory(config))
    return result
