"""Safe warm-JIT benchmark for the 3D cellular-automata engine.

Examples::

    python benchmarks/benchmark_3d.py
    python benchmarks/benchmark_3d.py --max-size 192 --steps 100

The benchmark measures the first compiled call separately from steady-state
calls. Sizes are attempted in increasing order and an estimated memory limit
prevents accidental large allocations.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from emergent.core3d.rules import parse_rule_3d
from emergent.core3d.simulate import run_steps_3d
from emergent.core3d.step import count_neighbors_3d

SIZES = (16, 32, 48, 64, 96, 128, 160, 192, 256)
RESULT_FIELDS = (
    "size",
    "shape",
    "cells",
    "device",
    "dtype",
    "estimated_grid_bytes",
    "estimated_working_set_bytes",
    "compile_ms",
    "warm_ms_per_generation",
    "warm_generations_per_second",
    "peak_memory_bytes",
    "status",
    "error",
)


def _benchmark_size(size: int, steps: int, repeats: int) -> dict[str, Any]:
    grid = jnp.zeros((size, size, size), dtype=jnp.uint8)
    grid = grid.at[size // 2, size // 2, size // 2].set(1)
    rule = parse_rule_3d("B6/S5,6,7")
    # Establish the neighbor implementation in the same process and device.
    count_neighbors_3d(grid).block_until_ready()

    compile_start = time.perf_counter()
    run_steps_3d(grid, rule, steps).block_until_ready()
    compile_ms = (time.perf_counter() - compile_start) * 1000

    warm_start = time.perf_counter()
    for _ in range(repeats):
        run_steps_3d(grid, rule, steps).block_until_ready()
    warm_elapsed = time.perf_counter() - warm_start
    generations = steps * repeats
    warm_ms_per_generation = warm_elapsed * 1000 / generations if generations else 0.0
    return {
        "size": size,
        "shape": f"{size}x{size}x{size}",
        "cells": size**3,
        "device": str(jax.devices()[0]),
        "dtype": "uint8",
        "estimated_grid_bytes": size**3,
        "estimated_working_set_bytes": size**3 * 30,
        "compile_ms": compile_ms,
        "warm_ms_per_generation": warm_ms_per_generation,
        "warm_generations_per_second": 1000 / warm_ms_per_generation
        if warm_ms_per_generation
        else 0.0,
        "peak_memory_bytes": "",
        "status": "ok",
        "error": "",
    }


def run_benchmark(
    *,
    max_size: int = 128,
    steps: int = 10,
    repeats: int = 3,
    max_memory_gb: float = 4.0,
    output_dir: str | Path = "artifacts/benchmarks",
) -> list[dict[str, Any]]:
    """Benchmark progressively larger cubes without intentionally exhausting memory."""

    if max_size < 1 or steps < 1 or repeats < 1 or max_memory_gb <= 0:
        raise ValueError("max_size, steps, repeats, and max_memory_gb must be positive")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    memory_limit = max_memory_gb * 1024**3
    for size in SIZES:
        if size > max_size:
            continue
        estimated_bytes = size**3
        estimated_working_set_bytes = estimated_bytes * 30
        if estimated_working_set_bytes > memory_limit:
            rows.append(
                {
                    "size": size,
                    "shape": f"{size}x{size}x{size}",
                    "cells": size**3,
                    "device": str(jax.devices()[0]),
                    "dtype": "uint8",
                    "estimated_grid_bytes": estimated_bytes,
                    "estimated_working_set_bytes": estimated_working_set_bytes,
                    "compile_ms": "",
                    "warm_ms_per_generation": "",
                    "warm_generations_per_second": "",
                    "peak_memory_bytes": "",
                    "status": "skipped-memory-limit",
                    "error": f"estimated working set exceeds {max_memory_gb:g} GiB limit",
                }
            )
            print(f"{size:>4}^3: SKIPPED (estimated grid exceeds memory limit)")
            continue
        try:
            row = _benchmark_size(size, steps, repeats)
        except Exception as exc:  # Report one failed size and continue safely.
            row = {
                "size": size,
                "shape": f"{size}x{size}x{size}",
                "cells": size**3,
                "device": str(jax.devices()[0]),
                "dtype": "uint8",
                "estimated_grid_bytes": estimated_bytes,
                "estimated_working_set_bytes": estimated_working_set_bytes,
                "compile_ms": "",
                "warm_ms_per_generation": "",
                "warm_generations_per_second": "",
                "peak_memory_bytes": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"{size:>4}^3: compile {row['compile_ms']:8.2f} ms | "
                f"warm {row['warm_ms_per_generation']:8.3f} ms/gen | "
                f"{row['warm_generations_per_second']:8.2f} gen/s"
            )
        else:
            print(f"{size:>4}^3: FAILED {row['error']}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = target / f"benchmark_3d_{stamp}.csv"
    json_path = target / f"benchmark_3d_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dimensions": 3,
        "sizes": [row["size"] for row in rows],
        "steps": steps,
        "repeats": repeats,
        "max_size": max_size,
        "max_memory_gb": max_memory_gb,
        "device": str(jax.devices()[0]),
        "timestamp": datetime.now(UTC).isoformat(),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-memory-gb", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()
    print(f"JAX device: {jax.devices()[0]}")
    run_benchmark(
        max_size=args.max_size,
        steps=args.steps,
        repeats=args.repeats,
        max_memory_gb=args.max_memory_gb,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
