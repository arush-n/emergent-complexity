"""Deterministic self-folding accessibility profiles.

The dynamic-programming portion is JAX compiled.  Only the small traceback
that converts an optimal pairing table into a boolean paired mask runs on the
host; species cache management is intentionally host-side because sequence
lengths are ragged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..canonical import ShapeKey
from .alphabet import CANONICAL_PAIRS, WOBBLE_PAIRS, validate_bases
from .sequence import ChemicalSequence


def _pair_matrix(bases: Any, allow_gu_wobble: bool) -> jax.Array:
    values = jnp.asarray(bases, dtype=jnp.uint8)
    first = values[:, None]
    second = values[None, :]
    canonical = ((first == 0) & (second == 3)) | ((first == 3) & (second == 0))
    canonical |= ((first == 1) & (second == 2)) | ((first == 2) & (second == 1))
    wobble = ((first == 2) & (second == 3)) | ((first == 3) & (second == 2))
    return canonical | (bool(allow_gu_wobble) & wobble)


@jax.jit(static_argnums=(1, 2))
def _nussinov_table(
    bases: jax.Array,
    minimum_hairpin_separation: int,
    allow_gu_wobble: bool,
) -> jax.Array:
    """Build a Nussinov maximum-pair table with static sequence shape."""

    length = bases.shape[0]
    pairable = _pair_matrix(bases, allow_gu_wobble)
    table = jnp.zeros((length, length), dtype=jnp.int16)
    indices = jnp.arange(length)

    def span_body(span: int, current: jax.Array) -> jax.Array:
        def start_body(start: int, table_value: jax.Array) -> jax.Array:
            end = start + span
            skip_start = table_value[start + 1, end]
            skip_end = table_value[start, end - 1]
            pair_value = table_value[start + 1, end - 1] + jnp.where(
                (end - start) > minimum_hairpin_separation,
                pairable[start, end],
                False,
            )
            split_end = jnp.minimum(indices + 1, length - 1)
            split_valid = (indices >= start + 1) & (indices < end)
            split_values = table_value[start, indices] + table_value[split_end, end]
            split_value = jnp.max(jnp.where(split_valid, split_values, -1))
            value = jnp.maximum(
                jnp.maximum(skip_start, skip_end), jnp.maximum(pair_value, split_value)
            )
            return table_value.at[start, end].set(value)

        return jax.lax.fori_loop(0, length - span, start_body, current)

    return jax.lax.fori_loop(1, length, span_body, table)


@jax.jit(static_argnums=(1, 2, 3))
def _windowed_pair_mask(
    bases: jax.Array,
    minimum_hairpin_separation: int,
    window: int,
    allow_gu_wobble: bool,
) -> jax.Array:
    """Return a bounded-window pairing opportunity mask for long sequences."""

    length = bases.shape[0]
    pairable = _pair_matrix(bases, allow_gu_wobble)
    indices = jnp.arange(length)
    distance = jnp.abs(indices[:, None] - indices[None, :])
    allowed = (distance > minimum_hairpin_separation) & (distance <= window)
    return jnp.any(pairable & allowed, axis=1)


def _trace_nussinov(
    table: np.ndarray,
    bases: np.ndarray,
    minimum_hairpin_separation: int,
    allow_gu_wobble: bool,
) -> set[tuple[int, int]]:
    """Trace one deterministic optimal pairing solution on the host."""

    length = bases.size
    paired: set[tuple[int, int]] = set()
    pairable = np.zeros((length, length), dtype=bool)
    for first in range(length):
        for second in range(length):
            first_base = int(bases[first])
            second_base = int(bases[second])
            pairable[first, second] = (first_base, second_base) in CANONICAL_PAIRS or (
                allow_gu_wobble and (first_base, second_base) in WOBBLE_PAIRS
            )

    def visit(start: int, end: int) -> None:
        if start >= end:
            return
        target = int(table[start, end])
        if target == int(table[start + 1, end]):
            visit(start + 1, end)
            return
        if target == int(table[start, end - 1]):
            visit(start, end - 1)
            return
        if end - start > minimum_hairpin_separation and pairable[start, end]:
            if target == int(table[start + 1, end - 1]) + 1:
                paired.add((start, end))
                visit(start + 1, end - 1)
                return
        for split in range(start + 1, end):
            if target == int(table[start, split]) + int(table[split + 1, end]):
                visit(start, split)
                visit(split + 1, end)
                return
        # The recurrence always has a matching branch.  This fallback keeps
        # malformed/short edge cases safe without changing valid results.
        visit(start + 1, end)

    if length > 1:
        visit(0, length - 1)
    return paired


def fold_pairs(
    bases: Any,
    *,
    minimum_hairpin_separation: int = 3,
    allow_gu_wobble: bool = True,
    max_nussinov_length: int = 256,
    window: int = 64,
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic non-crossing or windowed pairing set."""

    values = validate_bases(bases)
    if not isinstance(minimum_hairpin_separation, int) or minimum_hairpin_separation < 0:
        raise ValueError("minimum_hairpin_separation must be non-negative")
    if not isinstance(max_nussinov_length, int) or max_nussinov_length < 1:
        raise ValueError("max_nussinov_length must be positive")
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be positive")
    if not isinstance(allow_gu_wobble, bool):
        raise TypeError("allow_gu_wobble must be a boolean")
    if values.size <= max_nussinov_length:
        table = np.asarray(
            _nussinov_table(
                jnp.asarray(values),
                minimum_hairpin_separation,
                allow_gu_wobble,
            ),
            dtype=np.int16,
        )
        paired = _trace_nussinov(
            table,
            values,
            minimum_hairpin_separation,
            allow_gu_wobble,
        )
    else:
        paired_mask = np.asarray(
            _windowed_pair_mask(
                jnp.asarray(values),
                minimum_hairpin_separation,
                window,
                allow_gu_wobble,
            ),
            dtype=bool,
        )
        pairable = np.asarray(
            _pair_matrix(jnp.asarray(values), allow_gu_wobble),
            dtype=bool,
        )
        indices = np.arange(values.size)
        allowed = (np.abs(indices[:, None] - indices[None, :]) > minimum_hairpin_separation) & (
            np.abs(indices[:, None] - indices[None, :]) <= window
        )
        candidates = pairable & allowed & paired_mask[:, None] & paired_mask[None, :]
        paired = set()
        used: set[int] = set()
        for first in range(values.size):
            if first in used:
                continue
            partners = np.flatnonzero(candidates[first, first + 1 :]) + first + 1
            for second in partners.tolist():
                if second not in used:
                    paired.add((first, int(second)))
                    used.update((first, int(second)))
                    break
    return tuple(sorted(paired))


def compute_accessibility(
    bases: Any,
    *,
    mode: str = "simplified_fold",
    minimum_hairpin_separation: int = 3,
    paired_accessibility: float = 0.25,
    allow_gu_wobble: bool = True,
    max_nussinov_length: int = 256,
    window: int = 64,
) -> np.ndarray:
    """Return ``1`` for exposed bases and ``paired_accessibility`` otherwise."""

    values = validate_bases(bases)
    if mode == "none":
        result = np.ones(values.size, dtype=np.float64)
        result.setflags(write=False)
        return result
    if mode != "simplified_fold":
        raise ValueError("mode must be none or simplified_fold")
    if not 0.0 <= float(paired_accessibility) <= 1.0:
        raise ValueError("paired_accessibility must be between 0 and 1")
    pairs = fold_pairs(
        values,
        minimum_hairpin_separation=minimum_hairpin_separation,
        allow_gu_wobble=allow_gu_wobble,
        max_nussinov_length=max_nussinov_length,
        window=window,
    )
    result = np.ones(values.size, dtype=np.float64)
    for first, second in pairs:
        result[first] = float(paired_accessibility)
        result[second] = float(paired_accessibility)
    result.setflags(write=False)
    return result


@dataclass
class AccessibilityCache:
    """Cache one deterministic accessibility profile per exact morphology."""

    mode: str = "none"
    minimum_hairpin_separation: int = 3
    paired_accessibility: float = 0.25
    allow_gu_wobble: bool = True
    max_nussinov_length: int = 256
    window: int = 64
    profiles: dict[ShapeKey, np.ndarray] = field(default_factory=dict)

    def get(self, sequence: ChemicalSequence) -> np.ndarray:
        """Return and cache the profile for ``sequence.shape_key``."""

        profile = self.profiles.get(sequence.shape_key)
        if profile is None:
            profile = compute_accessibility(
                sequence.bases,
                mode=self.mode,
                minimum_hairpin_separation=self.minimum_hairpin_separation,
                paired_accessibility=self.paired_accessibility,
                allow_gu_wobble=self.allow_gu_wobble,
                max_nussinov_length=self.max_nussinov_length,
                window=self.window,
            )
            self.profiles[sequence.shape_key] = profile
        return profile


def accessibility_for_sequence(
    sequence: ChemicalSequence,
    *,
    mode: str = "simplified_fold",
    minimum_hairpin_separation: int = 3,
    paired_accessibility: float = 0.25,
    allow_gu_wobble: bool = True,
) -> np.ndarray:
    """Convenience wrapper for one sequence."""

    return compute_accessibility(
        sequence.bases,
        mode=mode,
        minimum_hairpin_separation=minimum_hairpin_separation,
        paired_accessibility=paired_accessibility,
        allow_gu_wobble=allow_gu_wobble,
    )
