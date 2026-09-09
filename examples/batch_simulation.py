"""Run many independent initial conditions through one Conway rule."""

import jax

from emergent.core.random import generate_random_grids
from emergent.core.rules import parse_rule
from emergent.core.simulate import generate_batched_trajectory


def main() -> None:
    keys = jax.random.split(jax.random.key(42), 100)
    initial_grids = generate_random_grids(keys, 128, 128, 0.20)
    trajectories = generate_batched_trajectory(
        initial_grids,
        parse_rule("B3/S23"),
        steps=50,
    )
    print(f"batched trajectory shape: {trajectories.shape}")


if __name__ == "__main__":
    main()
