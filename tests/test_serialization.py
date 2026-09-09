import numpy as np

from emergent.core.grid import empty_grid
from emergent.core.rules import parse_rule
from emergent.io.serialization import (
    export_state_json,
    import_state_json,
    load_grid,
    load_state,
    save_grid,
    save_state,
)


def test_json_state_round_trip() -> None:
    grid = empty_grid(5, 7).at[1, 2].set(1).at[4, 6].set(1)
    text = export_state_json(grid, parse_rule("B36/S23"), seed=42, density=0.2, generation=7)
    loaded = import_state_json(text)
    np.testing.assert_array_equal(loaded.grid, grid)
    assert loaded.rule == parse_rule("B36/S23")
    assert loaded.seed == 42
    assert loaded.generation == 7


def test_npy_npz_grid_and_state_round_trip(tmp_path) -> None:
    grid = empty_grid(4, 6).at[2, 3].set(1)
    npy_path = tmp_path / "grid.npy"
    npz_path = tmp_path / "grid.npz"
    state_path = tmp_path / "state.npz"
    save_grid(npy_path, grid)
    save_grid(npz_path, grid)
    save_state(state_path, grid, parse_rule("B3/S23"), generation=3)
    np.testing.assert_array_equal(load_grid(npy_path), grid)
    np.testing.assert_array_equal(load_grid(npz_path), grid)
    loaded = load_state(state_path)
    np.testing.assert_array_equal(loaded.grid, grid)
    assert loaded.generation == 3
