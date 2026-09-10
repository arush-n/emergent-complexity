"""Run a short, compile-aware performance probe for morphology interactions.

The probe compares a native CGOL control with active morphology interactions,
one full-batch and one split-batch configuration, and an optional less-frequent
morphology-detection configuration.  It uses one deterministic initial batch
for all comparable cases and excludes the one-time JAX compilation from the
reported steady-state timing.

Example::

    python -m emergent.experiments.morphology_interactions.benchmarks.short \
        --num-envs 128 --size 32 --steps 40 --warmup-steps 5 \
        --batch-size 64 --shared-universe-seed 42 \
        --output-dir artifacts/experiments/morphology_interactions/benchmarks/short_cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..config import MorphologyExperimentConfig
from ..engine import initial_grid_from_config
from ..parallel import run_parallel


def _initial_batch(config: MorphologyExperimentConfig, num_envs: int) -> np.ndarray:
    """Build one deterministic initial batch shared by comparable cases."""

    grids = [
        initial_grid_from_config(replace(config, seed=config.seed + index))
        for index in range(num_envs)
    ]
    return np.stack(grids, axis=0).astype(np.uint8, copy=False)


def _warm_compile(
    config: MorphologyExperimentConfig,
    initial_grids: np.ndarray,
    *,
    batch_size: int,
    pair_capacity: int,
    host_workers: int,
    shared_universe_seed: int | None,
) -> None:
    """Compile both the active detection and local-step paths before timing."""

    compile_config = replace(config, steps=max(1, config.warmup_steps + 1), warmup_steps=0)
    run_parallel(
        compile_config,
        num_envs=initial_grids.shape[0],
        initial_grids=initial_grids,
        batch_size=batch_size,
        host_workers=host_workers,
        pair_capacity=pair_capacity,
        shared_universe_seed=shared_universe_seed,
        collect_metrics=False,
    )


def _timed_case(
    label: str,
    config: MorphologyExperimentConfig,
    initial_grids: np.ndarray,
    *,
    batch_size: int,
    pair_capacity: int,
    host_workers: int,
    shared_universe_seed: int | None,
) -> dict[str, Any]:
    """Measure one steady-state lockstep run after a compile warmup."""

    _warm_compile(
        config,
        initial_grids,
        batch_size=batch_size,
        pair_capacity=pair_capacity,
        host_workers=host_workers,
        shared_universe_seed=shared_universe_seed,
    )
    start = time.perf_counter()
    result = run_parallel(
        config,
        num_envs=initial_grids.shape[0],
        initial_grids=initial_grids,
        batch_size=batch_size,
        host_workers=host_workers,
        pair_capacity=pair_capacity,
        shared_universe_seed=shared_universe_seed,
        collect_metrics=False,
    )
    elapsed = time.perf_counter() - start
    transitions = initial_grids.shape[0] * config.steps
    return {
        "label": label,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "num_envs": initial_grids.shape[0],
        "size": config.height,
        "steps": config.steps,
        "warmup_steps": config.warmup_steps,
        "batch_size": batch_size,
        "host_workers": host_workers,
        "shared_universe_seed": shared_universe_seed,
        "component_backend": config.component_backend,
        "detect_every": config.detect_every,
        "interactions_enabled": config.interactions_enabled,
        "elapsed_seconds": elapsed,
        "environments_per_second": initial_grids.shape[0] / elapsed,
        "transitions_per_second": transitions / elapsed,
        "final_alive_mean": float(result.final_grids.sum(axis=(1, 2)).mean()),
    }


def run_short_benchmark(
    *,
    num_envs: int = 128,
    size: int = 32,
    steps: int = 40,
    warmup_steps: int = 5,
    seed: int = 42,
    density: float = 0.10,
    alpha: float = 0.5,
    batch_size: int | None = None,
    pair_capacity: int = 256,
    component_backend: str = "auto",
    host_workers: int = 0,
    shared_universe_seed: int | None = None,
    include_detection_interval: bool = True,
) -> list[dict[str, Any]]:
    """Return a small performance matrix with deterministic initial states.

    ``detect_every=2`` is reported as a separate case because it changes the
    experiment's physics.  It is useful for profiling host-analysis savings,
    but must not be treated as a pure implementation optimization.
    """

    if not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if batch_size is None:
        batch_size = min(num_envs, 64)
    if not isinstance(batch_size, int) or not 1 <= batch_size <= num_envs:
        raise ValueError("batch_size must be between 1 and num_envs")
    if not isinstance(pair_capacity, int) or pair_capacity < 1:
        raise ValueError("pair_capacity must be a positive integer")
    if not isinstance(host_workers, int) or host_workers < 0:
        raise ValueError("host_workers must be a non-negative integer")
    if shared_universe_seed is not None and (
        not isinstance(shared_universe_seed, int) or shared_universe_seed < 0
    ):
        raise ValueError("shared_universe_seed must be a non-negative integer")
    base = MorphologyExperimentConfig(
        width=size,
        height=size,
        steps=steps,
        warmup_steps=warmup_steps,
        seed=seed,
        density=density,
        alpha=alpha,
        component_backend=component_backend,
    )
    initial_grids = _initial_batch(base, num_envs)
    cases: list[tuple[str, MorphologyExperimentConfig, int]] = [
        (
            "control_batched",
            replace(base, interactions_enabled=False),
            batch_size,
        ),
        ("active_batched", base, batch_size),
        ("active_full_batch", base, num_envs),
    ]
    if include_detection_interval:
        cases.append(
            (
                "active_detect_every_2",
                replace(base, detect_every=2),
                batch_size,
            )
        )
    cases.append(("active_one_environment", base, 1))

    rows = []
    for label, config, case_batch_size in cases:
        case_grids = initial_grids if case_batch_size != 1 else initial_grids[:1]
        rows.append(
            _timed_case(
                label,
                config,
                case_grids,
                batch_size=min(case_batch_size, case_grids.shape[0]),
                pair_capacity=pair_capacity,
                host_workers=host_workers,
                shared_universe_seed=shared_universe_seed,
            )
        )
    return rows


def _write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact JSON and CSV benchmark results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "morphology_interactions",
        "benchmark": "short",
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cases": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = tuple(rows[0].keys()) if rows else ()
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the short benchmark CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pair-capacity", type=int, default=256)
    parser.add_argument("--host-workers", type=int, default=0)
    parser.add_argument("--shared-universe-seed", type=int, default=None)
    parser.add_argument(
        "--component-backend",
        choices=("auto", "python", "scipy"),
        default="auto",
    )
    parser.add_argument("--no-detection-interval", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the probe and print a compact JSON result."""

    args = build_parser().parse_args(argv)
    rows = run_short_benchmark(
        num_envs=args.num_envs,
        size=args.size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        density=args.density,
        alpha=args.alpha,
        batch_size=args.batch_size,
        pair_capacity=args.pair_capacity,
        host_workers=args.host_workers,
        shared_universe_seed=args.shared_universe_seed,
        component_backend=args.component_backend,
        include_detection_interval=not args.no_detection_interval,
    )
    if args.output_dir is not None:
        _write_results(args.output_dir, rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
