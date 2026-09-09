"""Straightforward JSON/NPZ persistence for reproducible simulator states."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..core.grid import Grid
from ..core.rules import Rule, format_rule, parse_rule


@dataclass(frozen=True)
class SavedState:
    """A loaded state and the metadata needed to reproduce its setup."""

    grid: Grid
    rule: Rule
    seed: int | None
    density: float | None
    generation: int

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])


def state_to_dict(
    grid: Grid,
    rule: Rule,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> dict[str, Any]:
    """Convert a state into JSON-compatible metadata plus nested grid data."""

    values = np.asarray(jax.device_get(grid))
    if values.ndim != 2:
        raise ValueError("only single two-dimensional grids can be serialized")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    return {
        "rule": format_rule(rule),
        "width": int(values.shape[1]),
        "height": int(values.shape[0]),
        "seed": seed,
        "density": density,
        "generation": int(generation),
        "grid": (values != 0).astype(np.uint8).tolist(),
    }


def _dict_to_state(payload: dict[str, Any]) -> SavedState:
    required = {"rule", "grid"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"serialized state is missing fields: {', '.join(sorted(missing))}")
    grid = jnp.asarray(payload["grid"], dtype=jnp.uint8)
    if grid.ndim != 2:
        raise ValueError("serialized grid must be two-dimensional")
    generation = int(payload.get("generation", 0))
    if generation < 0:
        raise ValueError("generation must be non-negative")
    seed_value = payload.get("seed")
    density_value = payload.get("density")
    return SavedState(
        grid=(grid != 0).astype(jnp.uint8),
        rule=parse_rule(str(payload["rule"])),
        seed=None if seed_value is None else int(seed_value),
        density=None if density_value is None else float(density_value),
        generation=generation,
    )


def export_state_json(
    grid: Grid,
    rule: Rule,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> str:
    """Serialize a state as readable JSON text for browser export."""

    return json.dumps(
        state_to_dict(
            grid,
            rule,
            seed=seed,
            density=density,
            generation=generation,
        ),
        indent=2,
    )


def import_state_json(text: str) -> SavedState:
    """Load a state from JSON text."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid state JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("state JSON must contain an object")
    return _dict_to_state(payload)


def save_grid(path: str | Path, grid: Grid) -> None:
    """Save only a grid as ``.npy`` or ``.npz``."""

    target = Path(path)
    values = np.asarray(jax.device_get(grid), dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError("only single two-dimensional grids can be saved")
    if target.suffix.lower() == ".npy":
        np.save(target, values)
    elif target.suffix.lower() == ".npz":
        np.savez_compressed(target, grid=values)
    else:
        raise ValueError("grid paths must end in .npy or .npz")


def load_grid(path: str | Path) -> Grid:
    """Load a grid from ``.npy`` or ``.npz``."""

    target = Path(path)
    if target.suffix.lower() == ".npy":
        values = np.load(target, allow_pickle=False)
    elif target.suffix.lower() == ".npz":
        with np.load(target, allow_pickle=False) as archive:
            if "grid" not in archive:
                raise ValueError("NPZ file does not contain a 'grid' array")
            values = archive["grid"]
    else:
        raise ValueError("grid paths must end in .npy or .npz")
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("loaded grid must be two-dimensional")
    return jnp.asarray(values != 0, dtype=jnp.uint8)


def save_state(
    path: str | Path,
    grid: Grid,
    rule: Rule,
    *,
    seed: int | None = None,
    density: float | None = None,
    generation: int = 0,
) -> None:
    """Save a complete state as ``.json`` or metadata-bearing ``.npz``."""

    target = Path(path)
    payload = state_to_dict(
        grid,
        rule,
        seed=seed,
        density=density,
        generation=generation,
    )
    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif suffix == ".npz":
        metadata = {key: value for key, value in payload.items() if key != "grid"}
        values = np.asarray(payload["grid"], dtype=np.uint8)
        np.savez_compressed(target, grid=values, metadata=json.dumps(metadata))
    else:
        raise ValueError("complete states must end in .json or .npz")


def load_state(path: str | Path) -> SavedState:
    """Load a complete state from ``.json`` or metadata-bearing ``.npz``."""

    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".json":
        return import_state_json(target.read_text(encoding="utf-8"))
    if suffix != ".npz":
        raise ValueError("complete states must end in .json or .npz")
    with np.load(target, allow_pickle=False) as archive:
        if "grid" not in archive or "metadata" not in archive:
            raise ValueError("NPZ state must contain 'grid' and 'metadata'")
        metadata = json.loads(str(archive["metadata"].item()))
        metadata["grid"] = archive["grid"].tolist()
    return _dict_to_state(metadata)
