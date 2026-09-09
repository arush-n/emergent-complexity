import io
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from emergent.core3d.rules import parse_rule_3d
from emergent.io.serialization_3d import (
    export_state_json_3d,
    import_state_json_3d,
    load_grid_3d,
    load_state_3d,
    save_grid_3d,
    save_state_3d,
    state_to_dict_3d,
    state_to_npz_bytes_3d,
)


def test_3d_json_and_numpy_persistence(tmp_path: Path) -> None:
    grid = jnp.zeros((3, 4, 5), dtype=jnp.uint8).at[1, 2, 3].set(1)
    rule = parse_rule_3d("B6/S5,6,7")
    payload = state_to_dict_3d(grid, rule, seed=42, density=0.1, generation=8)
    loaded = import_state_json_3d(
        export_state_json_3d(grid, rule, seed=42, density=0.1, generation=8)
    )
    assert payload["dimensions"] == 3
    assert loaded.grid.shape == (3, 4, 5)
    assert loaded.generation == 8
    assert loaded.seed == 42
    assert loaded.rule == rule

    npy_path = tmp_path / "grid.npy"
    npz_path = tmp_path / "grid.npz"
    save_grid_3d(npy_path, grid)
    save_grid_3d(npz_path, grid)
    assert jnp.array_equal(load_grid_3d(npy_path), grid)
    assert jnp.array_equal(load_grid_3d(npz_path), grid)

    state_path = tmp_path / "state.npz"
    save_state_3d(state_path, grid, rule, seed=7, density=0.2, generation=4)
    restored = load_state_3d(state_path)
    assert jnp.array_equal(restored.grid, grid)
    assert restored.rule == rule
    assert restored.generation == 4

    content = state_to_npz_bytes_3d(grid, rule, generation=9)
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        assert jnp.array_equal(jnp.asarray(archive["grid"]), grid)
        assert json.loads(str(archive["metadata"].item()))["generation"] == 9
