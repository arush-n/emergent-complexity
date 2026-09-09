"""Run a deterministic Conway trajectory without starting the web app."""

import jax

from emergent.core.grid import empty_grid
from emergent.core.rules import parse_rule
from emergent.core.simulate import generate_trajectory
from emergent.io.patterns import GLIDER, place_pattern


def main() -> None:
    grid = place_pattern(empty_grid(32, 32), GLIDER, 4, 4)
    trajectory = generate_trajectory(grid, parse_rule("B3/S23"), steps=20)
    print(f"trajectory shape: {trajectory.shape}")
    print(f"JAX device: {jax.devices()[0]}")


if __name__ == "__main__":
    main()
