"""Exact, translation-aware canonical representations of Life morphologies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .components import MOORE_OFFSETS, Component


@dataclass(frozen=True, order=True)
class ShapeKey:
    """Exact identity for a canonical binary morphology.

    ``packed`` stores row-major bits with ``np.packbits(..., bitorder='big')``;
    ``height`` and ``width`` recover the meaningful bit count in the final
    byte.  No lossy vector is used for identity.
    """

    height: int
    width: int
    packed: bytes

    def to_bytes(self) -> bytes:
        """Serialize dimensions and packed bits with unambiguous boundaries."""

        return (
            int(self.height).to_bytes(4, "big", signed=False)
            + int(self.width).to_bytes(4, "big", signed=False)
            + len(self.packed).to_bytes(8, "big", signed=False)
            + self.packed
        )

    @property
    def packed_hex(self) -> str:
        """Return a compact, human-readable representation for metadata."""

        return self.packed.hex()


SINGLETON_SHAPE_KEY = ShapeKey(1, 1, b"\x80")


def shape_key_sort_key(key: ShapeKey) -> tuple[int, int, bytes]:
    """Return the canonical ordering token used for symmetric pairs."""

    return key.height, key.width, key.packed


def _validate_grid_shape(grid_shape: tuple[int, int] | None) -> tuple[int, int] | None:
    if grid_shape is None:
        return None
    if len(grid_shape) != 2 or any(not isinstance(value, int) for value in grid_shape):
        raise TypeError("grid_shape must be a (height, width) integer tuple")
    height, width = grid_shape
    if height <= 0 or width <= 0:
        raise ValueError("grid_shape dimensions must be positive")
    return height, width


def _coordinate_array(coordinates: Any) -> np.ndarray:
    if isinstance(coordinates, Component):
        values = coordinates.coordinates
    elif hasattr(coordinates, "coordinates"):
        values = coordinates.coordinates
    else:
        values = coordinates
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("coordinates must have shape (cell_count, 2) and be non-empty")
    if len({tuple(value) for value in values.tolist()}) != values.shape[0]:
        raise ValueError("coordinates must be unique")
    return values


def unwrap_toroidal_coordinates(
    coordinates: Any,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Unwrap a toroidal component into locally adjacent integer coordinates.

    The first world coordinate in row-major order becomes the origin.  BFS
    propagates the shortest Moore-neighbor displacement across each toroidal
    edge.  This makes cells at columns ``0`` and ``W-1`` (or rows ``0`` and
    ``H-1``) adjacent in the morphology rather than artificially far apart.
    """

    values = _coordinate_array(coordinates)
    height, width = _validate_grid_shape(grid_shape)  # type: ignore[misc]
    world = {tuple(value) for value in values.tolist()}
    root = min(world)
    return _unwrap_from_root(world, root, (height, width), values)


def _unwrap_from_root(
    world: set[tuple[int, int]],
    root: tuple[int, int],
    grid_shape: tuple[int, int],
    values: np.ndarray,
) -> np.ndarray:
    """Unwrap one component from one explicit toroidal root."""

    height, width = grid_shape
    unwrapped: dict[tuple[int, int], tuple[int, int]] = {root: (0, 0)}
    queue: deque[tuple[int, int]] = deque([root])

    while queue:
        current = queue.popleft()
        current_local = unwrapped[current]
        for row_delta, col_delta in MOORE_OFFSETS:
            neighbor = (
                (current[0] + row_delta) % height,
                (current[1] + col_delta) % width,
            )
            if neighbor not in world or neighbor in unwrapped:
                continue
            unwrapped[neighbor] = (
                current_local[0] + row_delta,
                current_local[1] + col_delta,
            )
            queue.append(neighbor)

    # A valid component from the detector is connected.  The check also gives
    # callers a clear error if they pass a disconnected coordinate collection.
    if len(unwrapped) != len(world):
        raise ValueError("coordinates must form one toroidal 8-connected component")
    return np.asarray([unwrapped[tuple(value)] for value in values.tolist()], dtype=np.int64)


def _trim_and_shift(coordinates: np.ndarray) -> np.ndarray:
    shifted = coordinates - coordinates.min(axis=0)
    height = int(shifted[:, 0].max()) + 1
    width = int(shifted[:, 1].max()) + 1
    matrix = np.zeros((height, width), dtype=np.uint8)
    matrix[shifted[:, 0], shifted[:, 1]] = 1
    return matrix


def _candidate_matrices(
    matrix: np.ndarray,
    *,
    rotation_invariant: bool,
    reflection_invariant: bool,
) -> list[np.ndarray]:
    rotations = range(4) if rotation_invariant else range(1)
    candidates: list[np.ndarray] = []
    for rotation in rotations:
        rotated = np.rot90(matrix, k=rotation).copy()
        candidates.append(rotated)
        if reflection_invariant:
            candidates.append(np.fliplr(rotated).copy())
    return candidates


def _has_toroidal_seam(world: set[tuple[int, int]], grid_shape: tuple[int, int]) -> bool:
    """Return whether a component contains an edge crossing a world seam."""

    height, width = grid_shape
    for row, col in world:
        for row_delta, col_delta in MOORE_OFFSETS:
            neighbor = (
                (row + row_delta) % height,
                (col + col_delta) % width,
            )
            if neighbor == (row, col) or neighbor not in world:
                continue
            if (neighbor[0] - row, neighbor[1] - col) != (row_delta, col_delta):
                return True
    return False


def _matrix_sort_key(matrix: np.ndarray) -> tuple[bytes, int, int]:
    packed = np.packbits(matrix.reshape(-1), bitorder="big").tobytes()
    return packed, int(matrix.shape[0]), int(matrix.shape[1])


def canonical_matrix(
    coordinates: Any,
    *,
    grid_shape: tuple[int, int] | None = None,
    rotation_invariant: bool = True,
    reflection_invariant: bool = False,
) -> np.ndarray:
    """Return the smallest trimmed canonical binary matrix for a component."""

    if not isinstance(rotation_invariant, bool) or not isinstance(reflection_invariant, bool):
        raise TypeError("rotation_invariant and reflection_invariant must be booleans")
    if reflection_invariant and not rotation_invariant:
        raise ValueError("reflection_invariant requires rotation_invariant")
    values = _coordinate_array(coordinates)
    shape = _validate_grid_shape(grid_shape)
    if shape is None:
        local_matrices = [_trim_and_shift(values)]
    else:
        # The minimum world coordinate is convenient for ordinary components,
        # but it is not translation-invariant when a component spans a torus
        # seam or an entire periodic dimension.  Enumerating all possible roots
        # makes the set of candidate local embeddings depend only on relative
        # toroidal adjacency, so translating the whole component cannot change
        # its identity.  The usual components are small, making this exact
        # correction inexpensive compared with the initial CPU flood fill.
        world = {tuple(value) for value in values.tolist()}
        if _has_toroidal_seam(world, shape):
            local_matrices = [
                _trim_and_shift(_unwrap_from_root(world, root, shape, values))
                for root in sorted(world)
            ]
        else:
            local_matrices = [_trim_and_shift(values)]
    candidates: list[np.ndarray] = []
    for matrix in local_matrices:
        candidates.extend(
            _candidate_matrices(
                matrix,
                rotation_invariant=rotation_invariant,
                reflection_invariant=reflection_invariant,
            )
        )
    return min(candidates, key=_matrix_sort_key).astype(np.uint8, copy=False)


def shape_key_from_matrix(matrix: Any) -> ShapeKey:
    """Create an exact key from a canonical binary matrix."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("canonical matrix must be a non-empty two-dimensional array")
    binary = (values != 0).astype(np.uint8)
    packed = np.packbits(binary.reshape(-1), bitorder="big").tobytes()
    return ShapeKey(int(binary.shape[0]), int(binary.shape[1]), packed)


def canonical_matrix_from_grid(
    grid: Any,
    *,
    rotation_invariant: bool = True,
    reflection_invariant: bool = False,
) -> np.ndarray:
    """Canonicalize a standalone binary morphology matrix."""

    values = np.asarray(grid)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("grid must be a non-empty two-dimensional matrix")
    coordinates = np.argwhere(values != 0)
    if coordinates.shape[0] == 0:
        raise ValueError("grid must contain at least one live cell")
    return canonical_matrix(
        coordinates,
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )


def canonicalize_component(
    component: Component | Any,
    *,
    grid_shape: tuple[int, int] | None = None,
    rotation_invariant: bool = True,
    reflection_invariant: bool = False,
) -> ShapeKey:
    """Canonicalize a component and return its exact ``ShapeKey``."""

    return shape_key_from_matrix(
        canonical_matrix(
            component,
            grid_shape=grid_shape,
            rotation_invariant=rotation_invariant,
            reflection_invariant=reflection_invariant,
        )
    )


def canonicalize(
    coordinates: Any,
    *,
    grid_shape: tuple[int, int] | None = None,
    rotation_invariant: bool = True,
    reflection_invariant: bool = False,
) -> ShapeKey:
    """Short alias for :func:`canonicalize_component`."""

    return canonicalize_component(
        coordinates,
        grid_shape=grid_shape,
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )


def canonicalize_grid(
    grid: Any,
    *,
    rotation_invariant: bool = True,
    reflection_invariant: bool = False,
) -> ShapeKey:
    """Canonicalize a standalone binary morphology and return its exact key."""

    return shape_key_from_matrix(
        canonical_matrix_from_grid(
            grid,
            rotation_invariant=rotation_invariant,
            reflection_invariant=reflection_invariant,
        )
    )


def matrix_from_shape_key(key: ShapeKey) -> np.ndarray:
    """Decode a packed exact key back into its canonical binary matrix."""

    if not isinstance(key, ShapeKey):
        raise TypeError("key must be a ShapeKey")
    bit_count = key.height * key.width
    unpacked = np.unpackbits(np.frombuffer(key.packed, dtype=np.uint8), bitorder="big")
    if unpacked.size < bit_count:
        raise ValueError("packed shape key does not contain enough bits")
    return unpacked[:bit_count].reshape((key.height, key.width)).astype(np.uint8)


def key_to_matrix(key: ShapeKey) -> np.ndarray:
    """Alias for :func:`matrix_from_shape_key`."""

    return matrix_from_shape_key(key)
