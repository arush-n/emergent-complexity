"""JSON and NPZ persistence for 3D states."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..core3d.grid import Grid3D
from ..core3d.rules import Rule3D, format_rule_3d, parse_rule_3d


@dataclass(frozen=True)
class SavedState3D:
    """A loaded 3D state and the metadata needed to reproduce its setup."""

    grid: Grid3D
    rule: Rule3D
    seed: int | None
    density: float | None
    generation: int

    @property
    def depth(self) -> int:
        return int(self.grid.shape[0])

    @property
    def height(self) -> int:
        return int(self.grid.shape[1])

    @property
    def width(self) -> int:
        return int(self.grid.shape[2])


def state_to_dict_3d(
    grid: Grid3D,
    rule: Rule3D,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> dict[str, Any]:
    """Convert a 3D state to JSON-compatible metadata and nested voxel data."""

    values = np.asarray(jax.device_get(grid))
    if values.ndim != 3:
        raise ValueError("only single three-dimensional grids can be serialized")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    return {
        "dimensions": 3,
        "rule": format_rule_3d(rule),
        "depth": int(values.shape[0]),
        "height": int(values.shape[1]),
        "width": int(values.shape[2]),
        "seed": seed,
        "density": density,
        "generation": int(generation),
        "grid": (values != 0).astype(np.uint8).tolist(),
    }


def _dict_to_state_3d(payload: dict[str, Any]) -> SavedState3D:
    if "rule" not in payload or "grid" not in payload:
        raise ValueError("serialized 3D state must contain 'rule' and 'grid'")
    grid = jnp.asarray(payload["grid"], dtype=jnp.uint8)
    if grid.ndim != 3:
        raise ValueError("serialized grid must be three-dimensional")
    expected_shape = tuple(
        int(payload[key]) for key in ("depth", "height", "width") if key in payload
    )
    if expected_shape and expected_shape != tuple(grid.shape):
        raise ValueError(f"serialized dimensions {expected_shape} do not match grid {grid.shape}")
    generation = int(payload.get("generation", 0))
    if generation < 0:
        raise ValueError("generation must be non-negative")
    seed_value = payload.get("seed")
    density_value = payload.get("density")
    return SavedState3D(
        grid=(grid != 0).astype(jnp.uint8),
        rule=parse_rule_3d(str(payload["rule"])),
        seed=None if seed_value is None else int(seed_value),
        density=None if density_value is None else float(density_value),
        generation=generation,
    )


def export_state_json_3d(
    grid: Grid3D,
    rule: Rule3D,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> str:
    """Serialize a complete 3D state as readable JSON text."""

    return json.dumps(
        state_to_dict_3d(
            grid,
            rule,
            seed=seed,
            density=density,
            generation=generation,
        ),
        indent=2,
    )


def state_to_npz_bytes_3d(
    grid: Grid3D,
    rule: Rule3D,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> bytes:
    """Serialize a complete 3D state to an in-memory NPZ download."""

    values = np.asarray(jax.device_get(grid), dtype=np.uint8)
    if values.ndim != 3:
        raise ValueError("only single three-dimensional grids can be serialized")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    metadata = {
        "dimensions": 3,
        "rule": format_rule_3d(rule),
        "depth": int(values.shape[0]),
        "height": int(values.shape[1]),
        "width": int(values.shape[2]),
        "seed": seed,
        "density": density,
        "generation": int(generation),
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        grid=(values != 0).astype(np.uint8),
        metadata=json.dumps(metadata),
    )
    return buffer.getvalue()


def import_state_json_3d(text: str) -> SavedState3D:
    """Load a 3D state from JSON text."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid 3D state JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("3D state JSON must contain an object")
    return _dict_to_state_3d(payload)


def save_grid_3d(path: str | Path, grid: Grid3D) -> None:
    """Save a single 3D grid as ``.npy`` or compressed ``.npz``."""

    target = Path(path)
    values = np.asarray(jax.device_get(grid), dtype=np.uint8)
    if values.ndim != 3:
        raise ValueError("only single three-dimensional grids can be saved")
    if target.suffix.lower() == ".npy":
        np.save(target, values)
    elif target.suffix.lower() == ".npz":
        np.savez_compressed(target, grid=values)
    else:
        raise ValueError("3D grid paths must end in .npy or .npz")


def load_grid_3d(path: str | Path) -> Grid3D:
    """Load a 3D grid from ``.npy`` or ``.npz``."""

    target = Path(path)
    if target.suffix.lower() == ".npy":
        values = np.load(target, allow_pickle=False)
    elif target.suffix.lower() == ".npz":
        with np.load(target, allow_pickle=False) as archive:
            if "grid" not in archive:
                raise ValueError("NPZ file does not contain a 'grid' array")
            values = archive["grid"]
    else:
        raise ValueError("3D grid paths must end in .npy or .npz")
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError("loaded grid must be three-dimensional")
    return jnp.asarray(values != 0, dtype=jnp.uint8)


def save_state_3d(
    path: str | Path,
    grid: Grid3D,
    rule: Rule3D,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> None:
    """Save a complete 3D state as ``.json`` or metadata-bearing ``.npz``."""

    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = state_to_dict_3d(
            grid,
            rule,
            seed=seed,
            density=density,
            generation=generation,
        )
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif target.suffix.lower() == ".npz":
        target.write_bytes(
            state_to_npz_bytes_3d(
                grid,
                rule,
                seed=seed,
                density=density,
                generation=generation,
            )
        )
    else:
        raise ValueError("complete 3D states must end in .json or .npz")


def load_state_3d(path: str | Path) -> SavedState3D:
    """Load a complete 3D state from ``.json`` or metadata-bearing ``.npz``."""

    target = Path(path)
    if target.suffix.lower() == ".json":
        return import_state_json_3d(target.read_text(encoding="utf-8"))
    if target.suffix.lower() != ".npz":
        raise ValueError("complete 3D states must end in .json or .npz")
    with np.load(target, allow_pickle=False) as archive:
        if "grid" not in archive or "metadata" not in archive:
            raise ValueError("3D NPZ state must contain 'grid' and 'metadata'")
        metadata = json.loads(str(archive["metadata"].item()))
        metadata["grid"] = archive["grid"].tolist()
    return _dict_to_state_3d(metadata)
