"""Run a large, auditable parallel morphology-interaction trial.

This probe is intentionally separate from the short throughput benchmark. It
uses one shared universe law across many independent initial grids, increases
the fixed-dimensional interaction space, and aggregates the resulting species,
pair, and local-rule diversity. It does not change the native simulator.

Example::

    python -m emergent.experiments.morphology_interactions.benchmarks.scale \
        --num-envs 512 --size 64 --steps 200 --warmup-steps 40 \
        --identity-dim 256 --max-rule-changes 6 --batch-size 128 \
        --shared-universe-seed 42 \
        --output-dir artifacts/experiments/morphology_interactions/benchmarks/scale_d256
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..config import MorphologyExperimentConfig
from ..encoding import audit_encodings
from ..engine import initial_grid_from_config
from ..parallel import ParallelExperimentResult, run_parallel


def _initial_batch(config: MorphologyExperimentConfig, num_envs: int) -> np.ndarray:
    """Build deterministic independent starting grids for all environments."""

    grids = [
        initial_grid_from_config(replace(config, seed=config.seed + index))
        for index in range(num_envs)
    ]
    return np.stack(grids, axis=0).astype(np.uint8, copy=False)


def _aggregate_generation_records(result: ParallelExperimentResult) -> list[dict[str, Any]]:
    """Aggregate per-environment metric rows without losing generation order."""

    by_generation: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for engine in result.engines:
        for row in engine.metrics.records:
            by_generation[int(row["generation"])].append(row)

    fields = (
        "alive_cells",
        "alive_fraction",
        "component_count",
        "unique_species_total",
        "species_observed_this_step",
        "new_species_this_step",
        "active_interaction_count",
        "unique_interaction_pairs_total",
        "unique_local_rules_total",
        "mean_component_size",
        "max_component_size",
        "mean_interaction_strength",
        "births",
        "deaths",
        "changed_cells",
    )
    rows: list[dict[str, Any]] = []
    for generation in sorted(by_generation):
        records = by_generation[generation]
        row: dict[str, Any] = {
            "generation": generation,
            "environments": len(records),
        }
        for field in fields:
            values = [float(record[field]) for record in records]
            row[f"{field}_mean"] = float(np.mean(values))
            if field in {
                "species_observed_this_step",
                "new_species_this_step",
                "active_interaction_count",
                "births",
                "deaths",
                "changed_cells",
            }:
                row[f"{field}_total"] = int(sum(int(value) for value in values))
        rows.append(row)
    return rows


def summarize_parallel_result(
    result: ParallelExperimentResult,
    *,
    shared_universe_seed: int,
) -> dict[str, Any]:
    """Return global diversity, scaling, and throughput measurements."""

    species_keys = set()
    pair_keys = set()
    rule_ids = set()
    per_environment: list[dict[str, Any]] = []
    for index, engine in enumerate(result.engines):
        species_keys.update(engine.species_registry)
        pair_keys.update(engine.interaction_cache)
        rule_ids.update(
            int(interaction.rule_id) for interaction in engine.interaction_cache.values()
        )
        endpoint = engine.metrics.summary(
            result.final_grids[index],
            engine.species_registry,
            engine.interaction_cache,
            last_generation=engine.generation,
        )
        per_environment.append(
            {
                "environment": index,
                "seed": result.seeds[index],
                **endpoint,
                "final_alive_cells": int(np.count_nonzero(result.final_grids[index])),
            }
        )

    elapsed = float(result.elapsed_seconds)
    transitions = len(result.engines) * result.config.steps
    encoding = audit_encodings(
        species_keys,
        universe_seed=shared_universe_seed,
        identity_dim=result.config.identity_dim,
    )
    return {
        "experiment": "morphology_interactions",
        "benchmark": "parallel_scale",
        "config": result.config.as_dict(),
        "parallel": {
            "num_envs": len(result.engines),
            "batch_size": result.config.width,
            "seeds": list(result.seeds),
            "shared_universe_seed": shared_universe_seed,
            "elapsed_seconds": elapsed,
            "environments_per_second": len(result.engines) / elapsed if elapsed else float("inf"),
            "transitions_per_second": transitions / elapsed if elapsed else float("inf"),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_version": getattr(jax, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "global_diversity": {
            "unique_species": len(species_keys),
            "unique_interaction_pairs": len(pair_keys),
            "unique_local_rules": len(rule_ids),
            "total_active_interactions": int(
                sum(int(row["total_active_interactions"]) for row in per_environment)
            ),
        },
        "encoding_audit": encoding,
        "endpoint_means": {
            "final_alive_cells": float(
                np.mean([row["final_alive_cells"] for row in per_environment])
            ),
            "unique_species": float(
                np.mean([row["unique_species_total"] for row in per_environment])
            ),
            "unique_pairs": float(
                np.mean([row["unique_interaction_pairs_total"] for row in per_environment])
            ),
            "unique_rules": float(
                np.mean([row["unique_local_rules_total"] for row in per_environment])
            ),
        },
        "per_environment": per_environment,
    }


def run_scale_trial(
    *,
    num_envs: int = 512,
    size: int = 64,
    steps: int = 200,
    warmup_steps: int = 40,
    seed: int = 42,
    density: float = 0.10,
    alpha: float = 0.5,
    identity_dim: int = 256,
    max_rule_changes: int = 6,
    interaction_threshold: float = 0.35,
    batch_size: int | None = 128,
    pair_capacity: int = 1024,
    host_workers: int = 0,
    shared_universe_seed: int = 42,
    component_backend: str = "scipy",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one reproducible high-dimensional parallel trial and write artifacts."""

    if not isinstance(shared_universe_seed, int) or shared_universe_seed < 0:
        raise ValueError("shared_universe_seed must be a non-negative integer")
    config = MorphologyExperimentConfig(
        width=size,
        height=size,
        seed=seed,
        density=density,
        steps=steps,
        warmup_steps=warmup_steps,
        alpha=alpha,
        identity_dim=identity_dim,
        max_rule_changes=max_rule_changes,
        interaction_threshold=interaction_threshold,
        component_backend=component_backend,
    )
    initial_grids = _initial_batch(config, num_envs)
    result = run_parallel(
        config,
        num_envs=num_envs,
        initial_grids=initial_grids,
        shared_universe_seed=shared_universe_seed,
        batch_size=batch_size,
        host_workers=host_workers,
        pair_capacity=pair_capacity,
        collect_metrics=True,
        record_snapshots=False,
    )
    summary = summarize_parallel_result(result, shared_universe_seed=shared_universe_seed)
    summary["parallel"]["batch_size"] = result.config.width if batch_size is None else batch_size
    if output_dir is not None:
        write_scale_artifacts(result, summary, output_dir)
    return summary


def write_scale_artifacts(
    result: ParallelExperimentResult,
    summary: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write the compact auditable bundle for one parallel scale trial."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "morphology_interactions",
        "benchmark": "parallel_scale",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": result.config.as_dict(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (target / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    rows = _aggregate_generation_records(result)
    fields = tuple(rows[0].keys()) if rows else ()
    with (target / "generations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    environment_rows = summary["per_environment"]
    environment_fields = tuple(environment_rows[0].keys()) if environment_rows else ()
    with (target / "environments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=environment_fields)
        if environment_fields:
            writer.writeheader()
            writer.writerows(environment_rows)
    np.savez_compressed(target / "final_grids.npz", grids=result.final_grids)
    return target


def build_parser() -> argparse.ArgumentParser:
    """Build the high-dimensional parallel-trial CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--identity-dim", type=int, default=256)
    parser.add_argument("--max-rule-changes", type=int, default=6)
    parser.add_argument("--interaction-threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-capacity", type=int, default=1024)
    parser.add_argument("--host-workers", type=int, default=0)
    parser.add_argument("--shared-universe-seed", type=int, default=42)
    parser.add_argument(
        "--component-backend",
        choices=("auto", "python", "scipy"),
        default="scipy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the high-dimensional parallel trial and print its summary."""

    args = build_parser().parse_args(argv)
    summary = run_scale_trial(
        num_envs=args.num_envs,
        size=args.size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        density=args.density,
        alpha=args.alpha,
        identity_dim=args.identity_dim,
        max_rule_changes=args.max_rule_changes,
        interaction_threshold=args.interaction_threshold,
        batch_size=args.batch_size,
        pair_capacity=args.pair_capacity,
        host_workers=args.host_workers,
        shared_universe_seed=args.shared_universe_seed,
        component_backend=args.component_backend,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "parallel": summary["parallel"],
                "global_diversity": summary["global_diversity"],
                "encoding_audit": summary["encoding_audit"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
