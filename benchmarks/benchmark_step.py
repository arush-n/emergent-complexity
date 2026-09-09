"""Small warm-JIT benchmark for one Conway generation.

Usage: ``python benchmarks/benchmark_step.py``
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from emergent.core.rules import parse_rule, rule_to_masks
from emergent.core.step import step_jit


def benchmark(size: int, generations: int) -> tuple[float, float]:
    grid = jnp.zeros((size, size), dtype=jnp.uint8).at[size // 2, size // 2].set(1)
    birth, survival = rule_to_masks(parse_rule("B3/S23"))
    grid = step_jit(grid, birth, survival)
    grid.block_until_ready()
    start = time.perf_counter()
    for _ in range(generations):
        grid = step_jit(grid, birth, survival)
    grid.block_until_ready()
    elapsed = time.perf_counter() - start
    milliseconds = elapsed * 1000 / generations
    return milliseconds, generations / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    args = parser.parse_args()
    print(f"JAX device: {jax.devices()[0]}")
    for size in (64, 128, 256, 512):
        milliseconds, generations_per_second = benchmark(size, args.generations)
        print(
            f"{size:>4} x {size:<4}: {milliseconds:8.3f} ms/generation  "
            f"{generations_per_second:8.2f} generations/sec"
        )


if __name__ == "__main__":
    main()
