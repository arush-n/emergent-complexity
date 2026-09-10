"""Deterministic four-symbol alphabet and RNA-style pairing primitives."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

BASES: tuple[str, ...] = ("A", "C", "G", "U")
BASE_TO_INT: dict[str, int] = {base: index for index, base in enumerate(BASES)}
INT_TO_BASE: dict[int, str] = dict(enumerate(BASES))

# A<->U and C<->G in the integer representation A=0, C=1, G=2, U=3.
COMPLEMENT: tuple[int, ...] = (3, 2, 1, 0)
CANONICAL_PAIRS: frozenset[tuple[int, int]] = frozenset({(0, 3), (3, 0), (1, 2), (2, 1)})
WOBBLE_PAIRS: frozenset[tuple[int, int]] = frozenset({(2, 3), (3, 2)})


def validate_bases(values: Any, *, copy: bool = True) -> np.ndarray:
    """Return a one-dimensional ``uint8`` array containing only bases 0--3."""

    if isinstance(values, str):
        try:
            result = np.asarray(
                [BASE_TO_INT[character] for character in values.upper()], dtype=np.uint8
            )
        except KeyError as exc:
            raise ValueError("sequence strings may contain only A, C, G, and U") from exc
    else:
        result = np.asarray(values, dtype=np.uint8)
        if result.ndim != 1:
            raise ValueError("bases must be one-dimensional")
        if copy:
            result = result.copy()
    if result.size == 0:
        raise ValueError("chemical sequences must be non-empty")
    if np.any(result > 3):
        raise ValueError("base values must lie in {0, 1, 2, 3}")
    return result


def bases_to_string(values: Any) -> str:
    """Convert integer bases to an ``A/C/G/U`` string."""

    bases = validate_bases(values)
    return "".join(INT_TO_BASE[int(value)] for value in bases)


def string_to_bases(value: str) -> np.ndarray:
    """Convert an ``A/C/G/U`` string to a read-only integer array."""

    result = validate_bases(value)
    result.setflags(write=False)
    return result


def complement_base(value: int) -> int:
    """Return the Watson--Crick complement of one integer base."""

    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError("base must be an integer") from exc
    if integer not in range(4):
        raise ValueError("base must lie in {0, 1, 2, 3}")
    return COMPLEMENT[integer]


def can_pair(first: int, second: int, *, allow_gu_wobble: bool = True) -> bool:
    """Return whether two bases can form a canonical or optional wobble pair."""

    pair = (int(first), int(second))
    return pair in CANONICAL_PAIRS or (allow_gu_wobble and pair in WOBBLE_PAIRS)


def pair_kind(first: int, second: int, *, allow_gu_wobble: bool = True) -> str:
    """Classify a pair as ``canonical``, ``wobble``, or ``mismatch``."""

    pair = (int(first), int(second))
    if pair in CANONICAL_PAIRS:
        return "canonical"
    if allow_gu_wobble and pair in WOBBLE_PAIRS:
        return "wobble"
    return "mismatch"


def reverse_complement(values: Any) -> np.ndarray:
    """Return the reverse-complement sequence as a read-only ``uint8`` array."""

    bases = validate_bases(values)
    result = np.asarray([COMPLEMENT[int(value)] for value in bases[::-1]], dtype=np.uint8)
    result.setflags(write=False)
    return result


def kmer_code(values: Any) -> int:
    """Encode a base sequence in base four using low-order-first powers."""

    bases = validate_bases(values)
    code = 0
    multiplier = 1
    for value in bases:
        code += int(value) * multiplier
        multiplier *= 4
    return code


def kmer_codes(values: Any, length: int) -> np.ndarray:
    """Return every contiguous k-mer code in deterministic sequence order."""

    bases = validate_bases(values)
    if not isinstance(length, int) or length < 1:
        raise ValueError("k-mer length must be a positive integer")
    if length > bases.size:
        return np.zeros(0, dtype=np.int64)
    count = bases.size - length + 1
    result = np.empty(count, dtype=np.int64)
    for start in range(count):
        result[start] = kmer_code(bases[start : start + length])
    return result


def decode_kmer(code: int, length: int) -> np.ndarray:
    """Decode a low-order-first base-four k-mer code."""

    if not isinstance(code, int) or code < 0:
        raise ValueError("k-mer code must be a non-negative integer")
    if not isinstance(length, int) or length < 1:
        raise ValueError("k-mer length must be a positive integer")
    values = np.empty(length, dtype=np.uint8)
    remainder = code
    for index in range(length):
        values[index] = remainder % 4
        remainder //= 4
    if remainder:
        raise ValueError("k-mer code does not fit the requested length")
    values.setflags(write=False)
    return values


def hamming_distance(first: Any, second: Any) -> int:
    """Return the Hamming distance between equal-length sequences."""

    left = validate_bases(first)
    right = validate_bases(second)
    if left.shape != right.shape:
        raise ValueError("Hamming distance requires equal-length sequences")
    return int(np.count_nonzero(left != right))


def sequence_entropy(values: Any) -> float:
    """Return Shannon entropy in bits for the four-symbol composition."""

    bases = validate_bases(values)
    counts = np.bincount(bases, minlength=4).astype(np.float64)
    probabilities = counts[counts > 0] / float(bases.size)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def base_frequencies(values: Any) -> tuple[float, float, float, float]:
    """Return normalized A/C/G/U frequencies in alphabet order."""

    bases = validate_bases(values)
    counts = np.bincount(bases, minlength=4).astype(np.float64)
    return tuple(float(value / bases.size) for value in counts)  # type: ignore[return-value]
