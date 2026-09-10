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
import numpy as np

from ...canonical import canonicalize_component
from ...components import detect_components
from ..chemistry import make_chemistry_universe
from ..config import RNAExperimentConfig
from ..engine import RNAChemistryEngine
from .runtime import ExactCycle, pack_grids, prepare, state_bytes, step_fields


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def shape_counts(grid: np.ndarray, config: RNAExperimentConfig) -> Counter:
    return Counter(
        canonicalize_component(
            component,
            grid_shape=grid.shape,
            rotation_invariant=config.rotation_invariant,
            reflection_invariant=config.reflection_invariant,
        )
        for component in detect_components(grid, backend=config.component_backend)
    )


def run_worker(args: argparse.Namespace) -> None:
    root = args.output_dir / f"worker_{args.worker_id:03d}"
    root.mkdir(parents=True, exist_ok=False)
    config = RNAExperimentConfig(
        width=args.size,
        height=args.size,
        seed=args.seed,
        steps=0,  # Scalar API field; continuous scheduling has no rollout horizon.
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
    universe = make_chemistry_universe(
        config.seed,
        beta=config.beta,
        calibration_threshold=config.interaction_threshold,
        calibration_size=config.calibration_size,
    )
    write_json(
        root / "manifest.json",
        {
            "config": config.as_dict(),
            "rollout_horizon": None,
            "stop_condition": "exact_full_state_recurrence",
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
    engines = []
    grids = []
    cycles = []
    baselines = []
    milestones = []
    trial_ids = []
    serial = 0
    # These caches contain immutable, species-only results under one chemistry
    # configuration. Repeated morphologies across worlds can reuse them safely.
    shared_sequences = {}
    shared_kmers = {}
    shared_accessibility = {}

    def spawn(slot: int) -> tuple:
        nonlocal serial
        trial = args.worker_id + serial * args.workers
        serial += 1
        initial_seed = args.seed + trial + 1
        grid = np.asarray(
            random_grid(
                key_from_seed(initial_seed),
                args.size,
                args.size,
                args.density,
            ),
            dtype=np.uint8,
        )
        engine = RNAChemistryEngine(config, initial_grid=grid, universe=universe)
        engine.sequence_cache = shared_sequences
        engine.kmer_index_cache = shared_kmers
        engine.accessibility_cache.profiles = shared_accessibility
        events.write(
            json.dumps(
                {"event": "start", "trial": trial, "slot": slot, "initial_seed": initial_seed}
            )
            + "\n"
        )
        return (
            engine,
            grid,
            ExactCycle(state_bytes(engine, grid)),
            shape_counts(grid, config),
            {},
            trial,
        )

    for slot in range(args.batch_size):
        engine, grid, cycle, baseline, milestone, trial = spawn(slot)
        engines.append(engine)
        grids.append(grid)
        cycles.append(cycle)
        baselines.append(baseline)
        milestones.append(milestone)
        trial_ids.append(trial)
    current = np.stack(grids)
    pending = []
    tick = 0
    completed = 0
    copy_leads = 0
    active_sites_total = 0
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

    try:
        while not stopped and (args.max_ticks is None or tick < args.max_ticks):
            prepared = [prepare(engine, grid) for engine, grid in zip(engines, current)]
            rules = np.stack([item[0] for item in prepared])
            active_sites_total += sum(item[1] for item in prepared)
            following = np.asarray(jax.device_get(step_fields(current, rules)), dtype=np.uint8)
            pending.append(
                (
                    list(trial_ids),
                    [e.generation for e in engines],
                    current.copy(),
                    rules,
                    following.copy(),
                )
            )
            tick += 1
            current = following.copy()
            for slot, engine in enumerate(engines):
                engine.generation += 1
                counts = shape_counts(current[slot], config)
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
                                    "shape": [key.height, key.width, key.packed.hex()],
                                }
                            )
                            + "\n"
                        )
                period = cycles[slot].observe(state_bytes(engine, current[slot]))
                if period is not None:
                    reason = (
                        "extinct"
                        if not np.any(current[slot])
                        else ("stable" if period == 1 else "repeat")
                    )
                    events.write(
                        json.dumps(
                            {
                                "event": "terminal",
                                "trial": trial_ids[slot],
                                "slot": slot,
                                "trace_tick": tick - 1,
                                "generation": engine.generation,
                                "period": period,
                                "reason": reason,
                            }
                        )
                        + "\n"
                    )
                    completed += 1
                    replacement = spawn(slot)
                    (
                        engines[slot],
                        current[slot],
                        cycles[slot],
                        baselines[slot],
                        milestones[slot],
                        trial_ids[slot],
                    ) = replacement
                elif len(engine.pair_cache) + len(engine.sequence_cache) > cache_limit:
                    # Pure memoization may be discarded; bound sites retain their physics.
                    engine.pair_cache.clear()
                    engine.sequence_cache.clear()
                    engine.kmer_index_cache.clear()
                    engine.species_registry.clear()
                    engine.accessibility_cache.profiles.clear()
            if len(pending) >= args.trace_chunk:
                flush()
            now = time.perf_counter()
            if now - last_status >= 5 or args.max_ticks == tick:
                write_json(
                    root / "status.json",
                    {
                        "running": True,
                        "ticks": tick,
                        "completed_trials": completed,
                        "started_trials": serial,
                        "active_worlds": args.batch_size,
                        "max_active_age": max(e.generation for e in engines),
                        "copy_growth_leads": copy_leads,
                        "active_sites_total": active_sites_total,
                        "elapsed_seconds": now - started,
                        "environment_steps_per_second": tick * args.batch_size / (now - started),
                    },
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
            {
                "running": False,
                "ticks": tick,
                "completed_trials": completed,
                "started_trials": serial,
                "active_worlds": args.batch_size,
                "max_active_age": max(e.generation for e in engines),
                "copy_growth_leads": copy_leads,
                "active_sites_total": active_sites_total,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        write_json(
            root / "stopped.json",
            {
                "ticks": tick,
                "completed_trials": completed,
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
