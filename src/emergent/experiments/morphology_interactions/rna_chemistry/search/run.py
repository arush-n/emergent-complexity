"""Continuously refill parallel RNA environments until interrupted.

Each world runs without an age limit, ending only on an exact full-state
recurrence. Independent process shards prepare chemistry while JAX advances
each shard in one fixed-shape batch. Every transition and applied rule field
is archived. Copy-growth events are leads requiring independent lineage tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ...canonical import ShapeKey, canonicalize_component
from ...components import Component, detect_components, detect_components_batch
from ..chemistry import make_chemistry_universe
from ..config import RNAExperimentConfig
from ..engine import RNAChemistryEngine
from .runtime import (
    ExactCycle,
    PreparedFields,
    grid_state_bytes,
    pack_grids,
    prepare_batch,
    state_bytes,
    step_fields,
    unchanged_batch,
)
from .schedule import TrialPlan, plan_for_trial, schedule_manifest


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def config_from_args(args: argparse.Namespace) -> RNAExperimentConfig:
    """Build the immutable base configuration shared by the trial scheduler."""

    return RNAExperimentConfig(
        width=args.size,
        height=args.size,
        seed=args.seed,
        steps=0,  # Continuous scheduling has no rollout horizon.
        alpha=args.alpha,
        density=args.density,
        component_backend=args.component_backend,
        binding_lifetime_mode=args.binding_lifetime_mode,
        accessibility_mode=args.accessibility_mode,
        warmup_steps=args.warmup_steps,
        detect_every=args.detect_every,
        interaction_radius=args.interaction_radius,
        binding_seed_length=args.binding_seed_length,
        minimum_binding_length=args.minimum_binding_length,
        site_max_rule_changes=args.site_max_rule_changes,
        interaction_threshold=args.interaction_threshold,
        interactions_enabled=not args.disable_interactions,
    )


def _component_keys(
    components: list[Component],
    config: RNAExperimentConfig,
    grid_shape: tuple[int, int],
    engine: RNAChemistryEngine | None = None,
) -> Counter[ShapeKey]:
    return Counter(
        engine._canonical_key(component)
        if engine is not None
        else canonicalize_component(
            component,
            grid_shape=grid_shape,
            rotation_invariant=config.rotation_invariant,
            reflection_invariant=config.reflection_invariant,
        )
        for component in components
    )


def shape_counts(grid: np.ndarray, config: RNAExperimentConfig) -> Counter[ShapeKey]:
    """Return exact morphology counts for compatibility with small callers."""

    components = detect_components(
        grid,
        min_component_cells=config.min_component_cells,
        backend=config.component_backend,
    )
    return _component_keys(components, config, grid.shape)


def shape_counts_and_components(
    grid: np.ndarray,
    config: RNAExperimentConfig,
    engine: RNAChemistryEngine | None = None,
) -> tuple[Counter[ShapeKey], list[Component]]:
    """Detect one grid once and return both telemetry and next-step evidence."""

    components = detect_components(
        grid,
        min_component_cells=config.min_component_cells,
        backend=config.component_backend,
    )
    return _component_keys(components, config, grid.shape, engine), components


def shape_counts_and_components_batch(
    grids: np.ndarray,
    configs: list[RNAExperimentConfig],
    engines: list[RNAChemistryEngine] | None = None,
) -> tuple[list[Counter[ShapeKey]], list[list[Component]]]:
    """Batch host-side component labeling while preserving per-trial settings."""

    values = np.asarray(grids, dtype=np.uint8)
    if values.ndim != 3 or len(configs) != values.shape[0]:
        raise ValueError("grids and configs must describe a matching batch")
    if engines is not None and len(engines) != len(configs):
        raise ValueError("engines and configs must describe a matching batch")
    groups: dict[tuple[int, str], list[int]] = {}
    for index, config in enumerate(configs):
        groups.setdefault((config.min_component_cells, config.component_backend), []).append(index)
    components_by_environment: list[list[Component] | None] = [None] * len(configs)
    counts: list[Counter[ShapeKey] | None] = [None] * len(configs)
    for (minimum_cells, backend), indices in groups.items():
        detected = detect_components_batch(
            values[indices],
            min_component_cells=minimum_cells,
            backend=backend,
        )
        for index, components in zip(indices, detected):
            components_by_environment[index] = components
            counts[index] = _component_keys(
                components,
                configs[index],
                values.shape[1:],
                None if engines is None else engines[index],
            )
    return (
        [item if item is not None else Counter() for item in counts],
        [item if item is not None else [] for item in components_by_environment],
    )


def _grid_key(grid: np.ndarray) -> str:
    """Return a stable identity for a complete initial state."""

    return hashlib.blake2b(
        np.ascontiguousarray(grid, dtype=np.uint8).tobytes(),
        digest_size=16,
        person=b"rna-start-v1",
    ).hexdigest()


def _shape_payload(engine: RNAChemistryEngine, key: ShapeKey) -> dict[str, object]:
    sequence = engine.sequence_cache.get(key)
    cells = np.unpackbits(np.frombuffer(key.packed, dtype=np.uint8))[: key.height * key.width]
    return {
        "height": key.height,
        "width": key.width,
        "packed": key.packed.hex(),
        "cells": int(cells.sum()),
        "sequence_length": None if sequence is None else sequence.length,
    }


def _pair_payload(pair: object) -> dict[str, object]:
    """Serialize stable pair-chemistry evidence for an interesting event."""

    chemistry = pair
    return {
        "species_a": [
            chemistry.species_a.height,
            chemistry.species_a.width,
            chemistry.species_a.packed.hex(),
        ],
        "species_b": [
            chemistry.species_b.height,
            chemistry.species_b.width,
            chemistry.species_b.packed.hex(),
        ],
        "candidate_seed_count": chemistry.candidate_seed_count,
        "raw_site_count": chemistry.raw_site_count,
        "successful_binding_sites": chemistry.successful_binding_sites,
        "best_binding_score": chemistry.best_binding_score,
        "total_paired_bases": chemistry.total_paired_bases,
        "distinct_site_rules": chemistry.distinct_site_rules,
        "motif_pairs": [list(site.motif.symmetric_codes) for site in chemistry.sites],
        "site_lengths": [site.site.length for site in chemistry.sites],
        "site_strengths": [site.strength for site in chemistry.sites],
    }


def run_worker(args: argparse.Namespace) -> None:
    root = args.output_dir / f"worker_{args.worker_id:03d}"
    root.mkdir(parents=True, exist_ok=False)
    base_config = config_from_args(args)
    universe = make_chemistry_universe(
        base_config.seed,
        beta=base_config.beta,
        calibration_threshold=base_config.interaction_threshold,
        calibration_size=base_config.calibration_size,
    )
    write_json(
        root / "manifest.json",
        {
            "config": base_config.as_dict(),
            "base_config": base_config.as_dict(),
            "strategy_schedule": schedule_manifest(base_config, args.strategy_schedule),
            "rollout_horizon": None,
            "stop_condition": (
                "exact_unchanged_grid_or_grid_recurrence_or_full_state_recurrence"
            ),
            "terminal_outcome": "failure_and_deterministic_slot_refill",
            "unchanged_grid_eviction": "exact_grid_equality_between_consecutive_states",
            "grid_recurrence_eviction": "exact_grid_only_recurrence_fast_path",
            "batch_size": args.batch_size,
            "worker_id": args.worker_id,
            "workers": args.workers,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "calibration": universe.calibration_report(),
            "rule_encoding": "bits 0..8 birth, bits 9..17 survival",
            "trace_chunk_ticks": args.trace_chunk,
            "trace_grid_encoding": "little-endian bitpacked; unpack with manifest height/width",
            "pid": os.getpid(),
            "initialization": "independent seeded random soups; shared universe seed",
            "copy_events": "population growth leads, not confirmed lineage replication",
            "trial_identity": "config_key + initial_state_key; duplicate test keys are skipped",
            "interesting_events": "first-seen exact shapes and chemistry pairs per worker/config",
        },
    )
    stopped = False

    def stop(_signal: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    from .....core.random import key_from_seed, random_grid

    events = (root / "events.jsonl").open("a", buffering=1)
    engines: list[RNAChemistryEngine] = []
    grids: list[np.ndarray] = []
    cycles: list[ExactCycle] = []
    grid_cycles: list[ExactCycle] = []
    baselines: list[Counter[ShapeKey]] = []
    milestones: list[dict[ShapeKey, int]] = []
    trial_ids: list[int] = []
    plans: list[TrialPlan] = []
    cached_components: list[list[Component] | None] = []
    serial = 0
    skipped_starts = 0
    seen_trial_keys: set[str] = set()
    seen_test_keys: set[str] = set()
    seen_shape_notes: set[tuple[str, ShapeKey]] = set()
    seen_interaction_notes: set[tuple[str, tuple[ShapeKey, ShapeKey]]] = set()
    # These pools contain deterministic species-only or pair-only results.
    # Pools are partitioned by every chemistry setting that can change their
    # result, so profiles can share work without sharing incompatible physics.
    shared_sequences: dict[tuple[object, ...], dict] = {}
    shared_kmers: dict[tuple[object, ...], dict] = {}
    shared_accessibility: dict[tuple[object, ...], dict] = {}
    shared_pairs: dict[str, dict] = {}

    def write_event(payload: dict[str, object]) -> None:
        events.write(json.dumps(payload, sort_keys=True) + "\n")

    def spawn(
        slot: int,
        *,
        previous_plan: TrialPlan | None = None,
        previous_failure_reason: str | None = None,
        bootstrap: bool = False,
    ) -> tuple:
        nonlocal serial, skipped_starts
        while True:
            trial = args.worker_id + serial * args.workers
            plan = plan_for_trial(
                base_config,
                trial=trial,
                schedule_index=serial,
                mode=args.strategy_schedule,
                previous_strategy=None if previous_plan is None else previous_plan.strategy.key,
            )
            serial += 1
            grid = np.asarray(
                random_grid(
                    key_from_seed(plan.initial_seed),
                    plan.config.height,
                    plan.config.width,
                    plan.config.density,
                ),
                dtype=np.uint8,
            )
            initial_state_key = _grid_key(grid)
            test_key = f"{plan.config_key}:{initial_state_key}"
            if test_key in seen_test_keys:
                skipped_starts += 1
                write_event(
                    {
                        "event": "skip_duplicate_start",
                        "trial": plan.trial,
                        "trial_key": plan.trial_key,
                        "strategy_key": plan.strategy.key,
                        "config_key": plan.config_key,
                        "initial_state_key": initial_state_key,
                        "test_key": test_key,
                    }
                )
                continue
            seen_trial_keys.add(plan.trial_key)
            seen_test_keys.add(test_key)
            break

        config = plan.config
        engine = RNAChemistryEngine(config, initial_grid=grid, universe=universe)
        sequence_pool_key = (
            config.sequence_mode,
            config.rotation_invariant,
            config.reflection_invariant,
        )
        kmer_pool_key = (config.binding_seed_length, config.allow_gu_wobble)
        accessibility_pool_key = (
            config.accessibility_mode,
            config.minimum_hairpin_separation,
            config.paired_accessibility,
            config.max_nussinov_length,
            config.fold_window,
            config.allow_gu_wobble,
        )
        engine.sequence_cache = shared_sequences.setdefault(sequence_pool_key, {})
        engine.kmer_index_cache = shared_kmers.setdefault(kmer_pool_key, {})
        engine.accessibility_cache.profiles = shared_accessibility.setdefault(
            accessibility_pool_key, {}
        )
        engine.pair_cache = shared_pairs.setdefault(plan.config_key, {})
        baseline: Counter[ShapeKey] = Counter()
        components: list[Component] | None = None
        if bootstrap:
            baseline, components = shape_counts_and_components(grid, config, engine)
        write_event(
            {
                "event": "start",
                "outcome": "active",
                "slot": slot,
                "replaces_trial": None if previous_plan is None else previous_plan.trial,
                "previous_failure_reason": previous_failure_reason,
                "initial_state_key": initial_state_key,
                "test_key": test_key,
                **plan.as_dict(),
            }
        )
        return (
            engine,
            grid,
            ExactCycle(state_bytes(engine, grid)),
            ExactCycle(grid_state_bytes(grid)),
            baseline,
            {},
            plan.trial,
            plan,
            components,
        )

    for slot in range(args.batch_size):
        replacement = spawn(slot)
        engine, grid, cycle, grid_cycle, baseline, milestone, trial, plan, components = replacement
        engines.append(engine)
        grids.append(grid)
        cycles.append(cycle)
        grid_cycles.append(grid_cycle)
        baselines.append(baseline)
        milestones.append(milestone)
        trial_ids.append(trial)
        plans.append(plan)
        cached_components.append(components)
    initial_counts, initial_components = shape_counts_and_components_batch(
        np.stack(grids), [engine.config for engine in engines], engines
    )
    for slot in range(args.batch_size):
        baselines[slot] = initial_counts[slot]
        cached_components[slot] = initial_components[slot]
    # The dense state remains on the JAX backend between transitions.  Host
    # copies are made only for ragged morphology/chemistry preparation,
    # archival, and exact-cycle bookkeeping.
    current = jnp.asarray(np.stack(grids), dtype=jnp.uint8)
    pending = []
    tick = 0
    completed = 0
    unchanged_evictions = 0
    grid_repeat_evictions = 0
    copy_leads = 0
    active_sites_total = 0
    interesting_shapes = 0
    interesting_interactions = 0
    started = time.perf_counter()
    last_status = started
    cache_limit = args.cache_limit

    def flush() -> None:
        if not pending:
            return
        target = root / f"trace_{tick - len(pending):012d}.npz"
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                trials=np.asarray([x[0] for x in pending], dtype=np.int64),
                ages=np.asarray([x[1] for x in pending], dtype=np.int64),
                before_bits=pack_grids(np.asarray([x[2] for x in pending], dtype=np.uint8)),
                rules=np.asarray([x[3] for x in pending], dtype=np.uint32),
                after_bits=pack_grids(np.asarray([x[4] for x in pending], dtype=np.uint8)),
                height=np.int32(args.size),
                width=np.int32(args.size),
            )
        temporary.replace(target)
        pending.clear()

    def annotate_preparation(slot: int, prepared: PreparedFields, trace_tick: int) -> None:
        nonlocal interesting_shapes, interesting_interactions
        engine = engines[slot]
        plan = plans[slot]
        for key in prepared.shape_keys:
            identity = (plan.config_key, key)
            if identity in seen_shape_notes or interesting_shapes >= args.interesting_limit:
                continue
            seen_shape_notes.add(identity)
            interesting_shapes += 1
            write_event(
                {
                    "event": "interesting_shape",
                    "outcome": "observation",
                    "trial": plan.trial,
                    "slot": slot,
                    "trace_tick": trace_tick,
                    "generation": engine.generation,
                    "strategy_key": plan.strategy.key,
                    "config_key": plan.config_key,
                    "shape": _shape_payload(engine, key),
                }
            )
        for chemistry in prepared.pair_chemistries:
            identity = (plan.config_key, chemistry.pair_key)
            if (
                identity in seen_interaction_notes
                or interesting_interactions >= args.interesting_limit
            ):
                continue
            seen_interaction_notes.add(identity)
            interesting_interactions += 1
            write_event(
                {
                    "event": "interesting_interaction",
                    "outcome": "observation",
                    "trial": plan.trial,
                    "slot": slot,
                    "trace_tick": trace_tick,
                    "generation": engine.generation,
                    "strategy_key": plan.strategy.key,
                    "config_key": plan.config_key,
                    "interaction": _pair_payload(chemistry),
                }
            )

    def status_payload(*, running: bool, elapsed: float) -> dict[str, object]:
        rate = tick * args.batch_size / elapsed if elapsed else 0.0
        return {
            "running": running,
            "ticks": tick,
            "completed_trials": completed,
            "terminal_failures": completed,
            "failed_trials": completed,
            "unchanged_evictions": unchanged_evictions,
            "grid_repeat_evictions": grid_repeat_evictions,
            "started_trials": serial,
            "skipped_duplicate_starts": skipped_starts,
            "unique_trial_keys": len(seen_trial_keys),
            "unique_test_keys": len(seen_test_keys),
            "active_worlds": args.batch_size,
            "max_active_age": max(e.generation for e in engines),
            "copy_growth_leads": copy_leads,
            "interesting_shapes": interesting_shapes,
            "interesting_interactions": interesting_interactions,
            "active_sites_total": active_sites_total,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": rate,
            "aggregate_environment_steps_per_second": rate,
        }

    try:
        while not stopped and (args.max_ticks is None or tick < args.max_ticks):
            host_before = np.asarray(jax.device_get(current), dtype=np.uint8)
            prepared = prepare_batch(
                engines,
                host_before,
                components_by_environment=cached_components,
            )
            for slot, details in enumerate(prepared):
                annotate_preparation(slot, details, tick)
            rules = np.stack([item.rules for item in prepared]).astype(np.uint32, copy=False)
            active_sites_total += sum(item.active_zone_count for item in prepared)
            following_device = step_fields(current, jnp.asarray(rules, dtype=jnp.uint32))
            following_device.block_until_ready()
            unchanged_device = unchanged_batch(current, following_device)
            unchanged_device.block_until_ready()
            unchanged_flags = np.asarray(jax.device_get(unchanged_device), dtype=bool)
            following = np.asarray(jax.device_get(following_device), dtype=np.uint8)
            pending.append(
                (
                    list(trial_ids),
                    [e.generation for e in engines],
                    host_before.copy(),
                    rules,
                    following.copy(),
                )
            )
            tick += 1
            current = following_device
            counts_after, components_after = shape_counts_and_components_batch(
                following,
                [engine.config for engine in engines],
                engines,
            )
            for slot, engine in enumerate(engines):
                engine.generation += 1
                counts = counts_after[slot]
                # This is deliberately a lead detector, not a replication
                # claim.  It watches every morphology, including species that
                # first appear after the initial soup, and links the event to
                # the exact archived transition carrying its applied rules.
                for key, count in counts.items():
                    baseline = baselines[slot].get(key, 0)
                    level = 4 if count >= 4 else 2 if count >= 2 else 0
                    if level > milestones[slot].get(key, 0):
                        milestones[slot][key] = level
                        copy_leads += 1
                        events.write(
                            json.dumps(
                                {
                                    "event": "copy_growth_lead",
                                    "trial": trial_ids[slot],
                                    "slot": slot,
                                    "trace_tick": tick - 1,
                                    "generation": engine.generation,
                                    "baseline": baseline,
                                    "copies": count,
                                    "level": level,
                                    "strategy_key": plans[slot].strategy.key,
                                    "config_key": plans[slot].config_key,
                                    "shape": _shape_payload(engine, key),
                                }
                            )
                            + "\n"
                        )
                period = cycles[slot].observe(state_bytes(engine, following[slot]))
                grid_period = grid_cycles[slot].observe(grid_state_bytes(following[slot]))
                unchanged = bool(unchanged_flags[slot])
                if unchanged or grid_period is not None or period is not None:
                    reason = (
                        "unchanged"
                        if unchanged
                        else (
                            "grid_repeat"
                            if grid_period is not None
                            else (
                                "extinct"
                                if not np.any(following[slot])
                                else ("stable" if period == 1 else "repeat")
                            )
                        )
                    )
                    write_event(
                        {
                            "event": "terminal",
                            "outcome": "failure",
                            "failure": True,
                            "trial": trial_ids[slot],
                            "slot": slot,
                            "trace_tick": tick - 1,
                            "generation": engine.generation,
                            "period": period,
                            "grid_period": grid_period,
                            "reason": reason,
                            "unchanged_grid": unchanged,
                            "strategy_key": plans[slot].strategy.key,
                            "config_key": plans[slot].config_key,
                            "trial_key": plans[slot].trial_key,
                            "initial_seed": plans[slot].initial_seed,
                            "initial_state_key": _grid_key(engine.initial_grid),
                        }
                    )
                    completed += 1
                    if unchanged:
                        unchanged_evictions += 1
                    if grid_period is not None:
                        grid_repeat_evictions += 1
                    replacement = spawn(
                        slot,
                        previous_plan=plans[slot],
                        previous_failure_reason=reason,
                        bootstrap=True,
                    )
                    (
                        engines[slot],
                        replacement_grid,
                        cycles[slot],
                        grid_cycles[slot],
                        baselines[slot],
                        milestones[slot],
                        trial_ids[slot],
                        plans[slot],
                        cached_components[slot],
                    ) = replacement
                    current = current.at[slot].set(jnp.asarray(replacement_grid, dtype=jnp.uint8))
                else:
                    cached_components[slot] = components_after[slot]
                if len(engine.pair_cache) + len(engine.sequence_cache) > cache_limit:
                    # Pure memoization may be discarded; bound sites retain their physics.
                    engine.pair_cache.clear()
                    engine.sequence_cache.clear()
                    engine.kmer_index_cache.clear()
                    engine.species_registry.clear()
                    engine._canonical_cache.clear()
                    engine.accessibility_cache.profiles.clear()
            if len(pending) >= args.trace_chunk:
                flush()
            now = time.perf_counter()
            if now - last_status >= 5 or args.max_ticks == tick:
                write_json(
                    root / "status.json",
                    status_payload(running=True, elapsed=now - started),
                )
                last_status = now
    finally:
        flush()
        events.write(
            json.dumps({"event": "interrupted", "ticks": tick, "active_trials": trial_ids}) + "\n"
        )
        events.close()
        write_json(
            root / "status.json",
            status_payload(running=False, elapsed=time.perf_counter() - started),
        )
        write_json(
            root / "stopped.json",
            {
                "ticks": tick,
                "completed_trials": completed,
                "terminal_failures": completed,
                "unchanged_evictions": unchanged_evictions,
                "grid_repeat_evictions": grid_repeat_evictions,
                "started_trials": serial,
                "skipped_duplicate_starts": skipped_starts,
                "reason": "error" if sys.exc_info()[0] else "operator_or_test_limit",
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    result.add_argument("--size", type=int, default=128)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--density", type=float, default=0.04)
    result.add_argument("--alpha", type=float, default=0.0)
    result.add_argument("--warmup-steps", type=int, default=0)
    result.add_argument("--detect-every", type=int, default=1)
    result.add_argument("--trace-chunk", type=int, default=4)
    result.add_argument("--cache-limit", type=int, default=4096)
    result.add_argument(
        "--strategy-schedule",
        choices=("fixed", "rotating"),
        default="rotating",
        help="use the CLI configuration for every trial or rotate deterministic profiles",
    )
    result.add_argument(
        "--interesting-limit",
        type=int,
        default=20_000,
        help="maximum first-seen shape and interaction notes per worker",
    )
    result.add_argument("--component-backend", choices=("python", "scipy", "auto"), default="scipy")
    result.add_argument("--accessibility-mode", choices=("none", "simplified_fold"), default="none")
    result.add_argument("--binding-lifetime-mode", choices=("instant", "energy"), default="energy")
    result.add_argument("--interaction-radius", type=int, default=2)
    result.add_argument("--binding-seed-length", type=int, default=4)
    result.add_argument("--minimum-binding-length", type=int, default=4)
    result.add_argument("--site-max-rule-changes", type=int, default=1)
    result.add_argument("--interaction-threshold", type=float, default=0.35)
    result.add_argument(
        "--disable-interactions",
        action="store_true",
        help="run the native Conway control while retaining the same trace format",
    )
    result.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="optional benchmark/operator limit, never a terminal classification",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    for name in (
        "workers",
        "batch_size",
        "size",
        "trace_chunk",
        "cache_limit",
        "detect_every",
        "interaction_radius",
        "binding_seed_length",
        "minimum_binding_length",
        "site_max_rule_changes",
        "interesting_limit",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_ticks is not None and args.max_ticks < 1:
        raise ValueError("max_ticks must be positive when supplied")
    if args.worker_id is not None:
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[4]
    source_hashes = {}
    with zipfile.ZipFile(args.output_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(source_root.rglob("*.py")):
            name = str(source.relative_to(source_root.parent))
            payload = source.read_bytes()
            source_hashes[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
    write_json(args.output_dir / "source_hashes.json", source_hashes)
    children = []

    def stop(_signal: int, _frame: object) -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write_json(
        args.output_dir / "launcher.json",
        {
            "pid": os.getpid(),
            "args": vars(args) | {"output_dir": str(args.output_dir)},
            "total_worlds": args.workers * args.batch_size,
        },
    )
    try:
        for worker in range(args.workers):
            with (args.output_dir / f"worker_{worker:03d}.log").open("w") as log:
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            __package__ + ".run",
                            *sys.argv[1:],
                            "--worker-id",
                            str(worker),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("worker failed; see worker logs")
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError("worker failed; see worker logs")
    finally:
        stop(0, None)
        for child in children:
            child.wait()


if __name__ == "__main__":
    main()
