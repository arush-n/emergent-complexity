"""Simulate HighLife from a random, reproducible initial condition."""

import jax

from emergent.core.random import random_grid
from emergent.core.rules import parse_rule
from emergent.core.simulate import run_steps


def main() -> None:
    grid = random_grid(jax.random.key(42), 64, 64, 0.20)
    final_grid = run_steps(grid, parse_rule("B36/S23"), steps=100)
    print(f"final shape: {final_grid.shape}; alive: {int(final_grid.sum())}")


if __name__ == "__main__":
    main()
