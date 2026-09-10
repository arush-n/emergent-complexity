"""Deterministic morphology-to-RNA-inspired sequence maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..canonical import ShapeKey, canonical_matrix_from_grid, matrix_from_shape_key
from .alphabet import validate_bases


@dataclass(frozen=True)
class ChemicalSequence:
    """A variable-length four-symbol interface derived from one ``ShapeKey``.

    ``site_coordinates`` are canonical ``(row, column)`` locations.  Header
    positions in ``exact_shape`` mode use ``(-1, -1)``; all local-surface
    positions correspond to reactive boundary cells.
    """

    shape_key: ShapeKey
    bases: np.ndarray
    site_coordinates: np.ndarray
    length: int
    mode: str = "local_surface"

    def __post_init__(self) -> None:
        bases = validate_bases(self.bases)
        coordinates = np.asarray(self.site_coordinates, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape != (bases.size, 2):
            raise ValueError("site_coordinates must have shape (length, 2)")
        if not isinstance(self.shape_key, ShapeKey):
            raise TypeError("shape_key must be a ShapeKey")
        if int(self.length) != bases.size:
            raise ValueError("length must equal the number of bases")
        if self.mode not in {"local_surface", "exact_shape"}:
            raise ValueError("mode must be local_surface or exact_shape")
        bases.setflags(write=False)
        coordinates.setflags(write=False)
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "site_coordinates", coordinates)
        object.__setattr__(self, "length", int(self.length))

    @property
    def text(self) -> str:
        """Return the human-readable ``A/C/G/U`` representation."""

        from .alphabet import bases_to_string

        return bases_to_string(self.bases)


def _canonical_input(shape: ShapeKey | Any) -> tuple[ShapeKey, np.ndarray]:
    if isinstance(shape, ShapeKey):
        return shape, matrix_from_shape_key(shape)
    matrix = canonical_matrix_from_grid(shape)
    from ..canonical import shape_key_from_matrix

    return shape_key_from_matrix(matrix), matrix


def _is_boundary(matrix: np.ndarray, row: int, col: int) -> bool:
    height, width = matrix.shape
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            if row_delta == 0 and col_delta == 0:
                continue
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if (
                neighbor_row < 0
                or neighbor_row >= height
                or neighbor_col < 0
                or neighbor_col >= width
                or matrix[neighbor_row, neighbor_col] == 0
            ):
                return True
    return False


def _local_surface_symbol(matrix: np.ndarray, row: int, col: int) -> int:
    """Map a 3x3 local structural descriptor to one nucleotide.

    The low bit is the parity of all eight neighboring live cells.  The high
    bit is the XOR of ``orthogonal>=2`` and ``diagonal>=2`` occupancy.  These
    are explicit local morphology properties, not hashes; a local edit only
    changes descriptors in its nearby neighborhood apart from canonical
    reorientation effects.
    """

    height, width = matrix.shape
    orthogonal = 0
    diagonal = 0
    total = 0
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            if row_delta == 0 and col_delta == 0:
                continue
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            occupied = int(
                0 <= neighbor_row < height
                and 0 <= neighbor_col < width
                and matrix[neighbor_row, neighbor_col] != 0
            )
            total += occupied
            if abs(row_delta) + abs(col_delta) == 1:
                orthogonal += occupied
            else:
                diagonal += occupied
    low_bit = total % 2
    high_bit = int((orthogonal >= 2) != (diagonal >= 2))
    return 2 * high_bit + low_bit


def local_surface_sequence(shape: ShapeKey | Any) -> ChemicalSequence:
    """Encode one canonical morphology boundary as a deterministic sequence."""

    key, matrix = _canonical_input(shape)
    coordinates = [
        (int(row), int(col))
        for row, col in np.argwhere(matrix != 0)
        if _is_boundary(matrix, int(row), int(col))
    ]
    if not coordinates:  # A full one-cell matrix is still a reactive surface.
        coordinates = [(int(row), int(col)) for row, col in np.argwhere(matrix != 0)]
    bases = np.asarray(
        [_local_surface_symbol(matrix, row, col) for row, col in coordinates],
        dtype=np.uint8,
    )
    return ChemicalSequence(
        key,
        bases,
        np.asarray(coordinates, dtype=np.int64),
        int(bases.size),
        mode="local_surface",
    )


def _uint32_base4(value: int) -> list[int]:
    if not 0 <= value < 2**32:
        raise ValueError("shape dimensions must fit in uint32")
    digits = [0] * 16
    remaining = value
    for index in range(15, -1, -1):
        digits[index] = remaining % 4
        remaining //= 4
    return digits


def _base4_uint32(digits: np.ndarray) -> int:
    result = 0
    for digit in digits:
        result = result * 4 + int(digit)
    return result


def exact_shape_sequence(shape: ShapeKey | Any) -> ChemicalSequence:
    """Encode dimensions and canonical cell bits into a reversible sequence."""

    key, matrix = _canonical_input(shape)
    flat = matrix.reshape(-1).astype(np.uint8)
    payload: list[int] = []
    for start in range(0, flat.size, 2):
        first = int(flat[start])
        second = int(flat[start + 1]) if start + 1 < flat.size else 0
        payload.append(2 * first + second)
    bases = np.asarray(
        _uint32_base4(key.height) + _uint32_base4(key.width) + payload, dtype=np.uint8
    )
    coordinates = np.full((bases.size, 2), -1, dtype=np.int64)
    for index, start in enumerate(range(0, flat.size, 2), start=32):
        coordinates[index] = np.asarray(np.unravel_index(start, matrix.shape), dtype=np.int64)
    return ChemicalSequence(key, bases, coordinates, int(bases.size), mode="exact_shape")


def sequence_from_shape_key(key: ShapeKey, *, mode: str = "local_surface") -> ChemicalSequence:
    """Create the configured sequence for an exact morphology key."""

    if mode == "local_surface":
        return local_surface_sequence(key)
    if mode == "exact_shape":
        return exact_shape_sequence(key)
    raise ValueError("mode must be local_surface or exact_shape")


def sequence_from_shape(shape: ShapeKey | Any, *, mode: str = "local_surface") -> ChemicalSequence:
    """Create a deterministic chemical sequence from a shape or grid."""

    return sequence_from_shape_key(_canonical_input(shape)[0], mode=mode)


def decode_exact_shape_sequence(sequence: ChemicalSequence | Any) -> ShapeKey:
    """Reverse ``exact_shape_sequence`` back to its authoritative ``ShapeKey``."""

    if isinstance(sequence, ChemicalSequence):
        bases = sequence.bases
        if sequence.mode != "exact_shape":
            raise ValueError("only exact_shape sequences are reversible")
    else:
        bases = validate_bases(sequence)
    if bases.size < 32:
        raise ValueError("exact-shape sequence is missing its dimension header")
    height = _base4_uint32(bases[:16])
    width = _base4_uint32(bases[16:32])
    if height < 1 or width < 1:
        raise ValueError("exact-shape sequence contains invalid dimensions")
    cell_count = height * width
    expected_payload = (cell_count + 1) // 2
    if bases.size != 32 + expected_payload:
        raise ValueError("exact-shape sequence has an invalid payload length")
    flat = np.zeros(cell_count, dtype=np.uint8)
    for index, value in enumerate(bases[32:]):
        flat[2 * index] = int(value) // 2
        if 2 * index + 1 < cell_count:
            flat[2 * index + 1] = int(value) % 2
    matrix = flat.reshape((height, width))
    if not np.any(matrix):
        raise ValueError("exact-shape sequence must contain a live cell")
    from ..canonical import shape_key_from_matrix

    decoded = shape_key_from_matrix(matrix)
    if isinstance(sequence, ChemicalSequence) and decoded != sequence.shape_key:
        raise ValueError("exact-shape sequence does not decode to its declared ShapeKey")
    return decoded


def boundary_coordinates(shape: ShapeKey | Any) -> np.ndarray:
    """Return canonical local-surface coordinates in sequence order."""

    return local_surface_sequence(shape).site_coordinates.copy()


def morphology_perimeter(shape: ShapeKey | Any) -> int:
    """Return the four-neighbor perimeter of a canonical morphology."""

    _, matrix = _canonical_input(shape)
    height, width = matrix.shape
    perimeter = 0
    for row, col in np.argwhere(matrix != 0):
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
    return perimeter
