"""Fixed-dimensional deterministic morphology encodings."""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canonical import ShapeKey, matrix_from_shape_key

UINT64_SCALE = float(2**64)
MORPHOLOGY_STATISTICS = (
    "cell_count",
    "bounding_box_density",
    "aspect_ratio",
    "normalized_perimeter",
    "horizontal_symmetry",
    "vertical_symmetry",
)


def _seed_bytes(seed: int) -> bytes:
    try:
        integer = operator.index(seed)
    except TypeError as exc:
        raise TypeError("universe seed must be an integer") from exc
    return (integer & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)


def deterministic_uint64(seed: int, namespace: str, index: int = 0) -> int:
    """Derive a stable 64-bit word without process-randomized Python hashing."""

    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    if not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    digest = hashlib.blake2b(digest_size=8, person=b"morph-int-v1")
    digest.update(_seed_bytes(seed))
    namespace_bytes = namespace.encode("utf-8")
    digest.update(len(namespace_bytes).to_bytes(4, "little"))
    digest.update(namespace_bytes)
    digest.update(index.to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest(), "little", signed=False)


def deterministic_uniform(
    seed: int,
    namespace: str,
    count: int,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """Return reproducible uniform values from a seed/namespace stream."""

    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise ValueError("low and high must be finite with high >= low")
    words = np.asarray(
        [deterministic_uint64(seed, namespace, index) for index in range(count)],
        dtype=np.uint64,
    )
    values = words.astype(np.float64) / UINT64_SCALE
    return low + (high - low) * values


def _as_canonical_matrix(shape: ShapeKey | Any) -> np.ndarray:
    if isinstance(shape, ShapeKey):
        matrix = matrix_from_shape_key(shape)
    else:
        matrix = np.asarray(shape)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("shape must be a non-empty two-dimensional matrix")
        matrix = (matrix != 0).astype(np.uint8)
    if not np.any(matrix):
        raise ValueError("shape must contain at least one live cell")
    return matrix


def morphology_statistics(shape: ShapeKey | Any) -> np.ndarray:
    """Return explicit morphology statistics in the documented order."""

    matrix = _as_canonical_matrix(shape)
    height, width = matrix.shape
    cell_count = int(matrix.sum())
    perimeter = 0
    for row, col in np.argwhere(matrix):
        for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbor_row = int(row) + row_delta
            neighbor_col = int(col) + col_delta
            if (
                neighbor_row < 0
                or neighbor_row >= height
                or neighbor_col < 0
                or neighbor_col >= width
                or matrix[neighbor_row, neighbor_col] == 0
            ):
                perimeter += 1
    horizontal_symmetry = float(np.mean(matrix == np.fliplr(matrix)))
    vertical_symmetry = float(np.mean(matrix == np.flipud(matrix)))
    return np.asarray(
        [
            float(cell_count),
            cell_count / float(height * width),
            width / float(height),
            perimeter / float(4 * cell_count),
            horizontal_symmetry,
            vertical_symmetry,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MorphologyEncoder:
    """A universe-fixed Fourier/statistics encoding of canonical shapes."""

    universe_seed: int
    identity_dim: int = 32
    frequency_span: float = 3.0
    frequency_x: np.ndarray = field(init=False, repr=False)
    frequency_y: np.ndarray = field(init=False, repr=False)
    phase: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity_dim, int) or self.identity_dim < len(MORPHOLOGY_STATISTICS):
            raise ValueError(f"identity_dim must be at least {len(MORPHOLOGY_STATISTICS)}")
        if not isinstance(self.frequency_span, (int, float)) or float(self.frequency_span) <= 0:
            raise ValueError("frequency_span must be positive")
        fourier_dim = self.identity_dim - len(MORPHOLOGY_STATISTICS)
        x = deterministic_uniform(
            self.universe_seed,
            "encoder-frequency-x",
            fourier_dim,
            low=-float(self.frequency_span),
            high=float(self.frequency_span),
        )
        y = deterministic_uniform(
            self.universe_seed,
            "encoder-frequency-y",
            fourier_dim,
            low=-float(self.frequency_span),
            high=float(self.frequency_span),
        )
        phase = deterministic_uniform(
            self.universe_seed,
            "encoder-phase",
            fourier_dim,
            low=0.0,
            high=2.0 * np.pi,
        )
        for value in (x, y, phase):
            value.setflags(write=False)
        object.__setattr__(self, "frequency_x", x)
        object.__setattr__(self, "frequency_y", y)
        object.__setattr__(self, "phase", phase)

    def encode(self, shape: ShapeKey | Any) -> np.ndarray:
        """Encode a canonical matrix/key into a normalized ``identity_dim`` vector."""

        matrix = _as_canonical_matrix(shape)
        rows, cols = np.nonzero(matrix)
        height, width = matrix.shape
        normalized_x = (cols.astype(np.float64) + 0.5) / width - 0.5
        normalized_y = (rows.astype(np.float64) + 0.5) / height - 0.5
        if self.frequency_x.size:
            fourier = np.cos(
                2.0
                * np.pi
                * (
                    self.frequency_x[:, None] * normalized_x[None, :]
                    + self.frequency_y[:, None] * normalized_y[None, :]
                )
                + self.phase[:, None]
            ).sum(axis=1) / np.sqrt(float(len(rows)))
        else:
            fourier = np.zeros(0, dtype=np.float64)
        vector = np.concatenate((fourier, morphology_statistics(matrix)))
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros(self.identity_dim, dtype=np.float64)
        return (vector / norm).astype(np.float64, copy=False)


def make_encoder(universe_seed: int, identity_dim: int = 32) -> MorphologyEncoder:
    """Construct the fixed-frequency encoder for one universe."""

    return MorphologyEncoder(universe_seed=universe_seed, identity_dim=identity_dim)


def encode_shape(
    shape: ShapeKey | Any,
    *,
    universe_seed: int,
    identity_dim: int = 32,
) -> np.ndarray:
    """One-shot convenience wrapper around :class:`MorphologyEncoder`."""

    return MorphologyEncoder(universe_seed, identity_dim).encode(shape)
