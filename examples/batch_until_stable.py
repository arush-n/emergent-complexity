"""Run many initial conditions in parallel until each reaches a fixed point."""

import jax

from emergent.core.random import generate_random_grids, key_from_seed
from emergent.core.rules import parse_rule
from emergent.core.simulate import run_batched_until_stable


def main() -> None:
    keys = jax.random.split(key_from_seed(42), 32)
    grids = generate_random_grids(keys, height=64, width=64, density=0.20)
    finals, steps_taken, settled = run_batched_until_stable(
        grids,
        parse_rule("B3/S23"),
        max_steps=500,
    )
    print(f"final batch shape: {finals.shape}")
    print(f"settled: {int(settled.sum())}/{settled.shape[0]}")
    print(f"maximum steps used: {int(steps_taken.max())}")


if __name__ == "__main__":
    main()
