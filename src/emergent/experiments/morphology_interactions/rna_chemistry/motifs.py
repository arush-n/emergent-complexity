"""Reaction motifs and fixed-size JAX feature construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .alphabet import kmer_code, sequence_entropy, validate_bases
from .binding import BindingSite

# 16 base-pair frequencies, 256 adjacent-pair frequencies, four pair-class
# fractions, length, two sequence entropies, and 256 one-hot features for each
# of the two sorted four-base context motifs.
MOTIF_FEATURE_DIM = 16 + 256 + 4 + 1 + 2 + 256 + 256
_DEFAULT_KERNEL_BATCH_CAPACITY = 32
_DEFAULT_KERNEL_LENGTH_CAPACITY = 32


def _bucket_size(value: int) -> int:
    """Return a power-of-two shape bucket for a positive segment length."""

    if value < 1:
        raise ValueError("bucketed dimensions must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class ReactionMotif:
    """Fixed-length flanking context associated with one binding site."""

    bases_a: tuple[int, ...]
    bases_b: tuple[int, ...]
    motif_length: int

    def __post_init__(self) -> None:
        if self.motif_length < 1:
            raise ValueError("motif_length must be positive")
        if len(self.bases_a) != self.motif_length or len(self.bases_b) != self.motif_length:
            raise ValueError("motif bases must match motif_length")
        if any(int(value) not in range(4) for value in (*self.bases_a, *self.bases_b)):
            raise ValueError("motif bases must lie in {0, 1, 2, 3}")

    @property
    def code_a(self) -> int:
        """Return the base-four code of the A-side context."""

        return kmer_code(np.asarray(self.bases_a, dtype=np.uint8))

    @property
    def code_b(self) -> int:
        """Return the base-four code of the B-side context."""

        return kmer_code(np.asarray(self.bases_b, dtype=np.uint8))

    @property
    def symmetric_codes(self) -> tuple[int, int]:
        """Return a stable unordered motif-pair identity."""

        return tuple(sorted((self.code_a, self.code_b)))  # type: ignore[return-value]


def _profile(values: Any, length: int) -> np.ndarray:
    if values is None:
        return np.ones(length, dtype=np.float64)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError("accessibility must match its sequence length")
    return result


def _context(
    bases: np.ndarray,
    start: int,
    end: int,
    length: int,
    accessibility: np.ndarray,
) -> tuple[int, ...]:
    """Collect nearest exposed flanking bases, then deterministic fallbacks."""

    candidates: list[int] = []
    for position in range(start - 1, -1, -1):
        if accessibility[position] > 0.5:
            candidates.append(int(bases[position]))
        if len(candidates) >= length:
            break
    if len(candidates) < length:
        for position in range(end, bases.size):
            if accessibility[position] > 0.5:
                candidates.append(int(bases[position]))
            if len(candidates) >= length:
                break
    # Short sequences can have no flank.  The binding segment is a useful
    # deterministic structural fallback and keeps the motif vocabulary alive.
    if len(candidates) < length:
        for position in range(start, end):
            candidates.append(int(bases[position]))
            if len(candidates) >= length:
                break
    candidates.extend([0] * (length - len(candidates)))
    return tuple(candidates[:length])


def extract_reaction_motif(
    bases_a: Any,
    bases_b: Any,
    site: BindingSite,
    *,
    motif_length: int = 4,
    accessibility_a: Any = None,
    accessibility_b: Any = None,
) -> ReactionMotif:
    """Extract deterministic exposed/flanking motif contexts around a site."""

    first = validate_bases(bases_a)
    second = validate_bases(bases_b)
    if not isinstance(motif_length, int) or motif_length < 1:
        raise ValueError("motif_length must be positive")
    if site.end_a > first.size or site.end_b > second.size:
        raise ValueError("binding site lies outside the supplied sequences")
    first_accessibility = _profile(accessibility_a, first.size)
    second_accessibility = _profile(accessibility_b, second.size)
    return ReactionMotif(
        _context(first, site.start_a, site.end_a, motif_length, first_accessibility),
        _context(second, site.start_b, site.end_b, motif_length, second_accessibility),
        motif_length,
    )


def _motif_features_kernel(
    segment_a: jax.Array,
    segment_b: jax.Array,
    motif_a: jax.Array,
    motif_b: jax.Array,
    valid_length: jax.Array,
) -> jax.Array:
    """Construct a symmetric reaction feature vector on the accelerator."""

    first = jnp.asarray(segment_a, dtype=jnp.uint8)
    second = jnp.asarray(segment_b, dtype=jnp.uint8)
    length = jnp.asarray(valid_length, dtype=jnp.int32)
    valid = jnp.arange(first.size, dtype=jnp.int32) < length
    pair_codes = first * 4 + second
    reverse_pair_codes = second * 4 + first
    pair_weights = valid.astype(jnp.float32)
    pair_counts = jnp.bincount(
        pair_codes,
        weights=pair_weights,
        length=16,
        minlength=16,
    ).astype(jnp.float32)
    reverse_counts = jnp.bincount(
        reverse_pair_codes,
        weights=pair_weights,
        length=16,
        minlength=16,
    ).astype(jnp.float32)
    pair_frequency = 0.5 * (pair_counts + reverse_counts) / jnp.maximum(length, 1)

    adjacent_codes = pair_codes[:-1] * 16 + pair_codes[1:]
    adjacent_reverse_codes = reverse_pair_codes[:-1] * 16 + reverse_pair_codes[1:]
    adjacent_valid = valid[:-1] & valid[1:]
    adjacent_weights = adjacent_valid.astype(jnp.float32)
    stack_counts = jnp.bincount(
        adjacent_codes,
        weights=adjacent_weights,
        length=256,
        minlength=256,
    ).astype(jnp.float32)
    reverse_stack_counts = jnp.bincount(
        adjacent_reverse_codes,
        weights=adjacent_weights,
        length=256,
        minlength=256,
    ).astype(jnp.float32)
    stack_frequency = 0.5 * (stack_counts + reverse_stack_counts) / jnp.maximum(length - 1, 1)

    gc = ((first == 2) & (second == 1)) | ((first == 1) & (second == 2))
    au = ((first == 0) & (second == 3)) | ((first == 3) & (second == 0))
    gu = ((first == 2) & (second == 3)) | ((first == 3) & (second == 2))
    mismatch = ~(gc | au | gu)
    pair_fractions = jnp.asarray(
        [
            jnp.sum(gc & valid) / jnp.maximum(length, 1),
            jnp.sum(au & valid) / jnp.maximum(length, 1),
            jnp.sum(gu & valid) / jnp.maximum(length, 1),
            jnp.sum(mismatch & valid) / jnp.maximum(length, 1),
        ],
        dtype=jnp.float32,
    )
    bounded_length = length.astype(jnp.float32) / (length.astype(jnp.float32) + 1.0)

    def entropy(values: jax.Array) -> jax.Array:
        counts = jnp.bincount(
            values,
            weights=pair_weights,
            length=4,
            minlength=4,
        ).astype(jnp.float32)
        probabilities = counts / jnp.maximum(length, 1)
        positive = jnp.where(probabilities > 0, probabilities, 1.0)
        return -jnp.sum(jnp.where(probabilities > 0, probabilities * jnp.log2(positive), 0.0)) / 2.0

    first_motif = jnp.asarray(motif_a, dtype=jnp.uint8)
    second_motif = jnp.asarray(motif_b, dtype=jnp.uint8)
    first_code = jnp.sum(first_motif * (4 ** jnp.arange(first_motif.size, dtype=jnp.uint32)))
    second_code = jnp.sum(second_motif * (4 ** jnp.arange(second_motif.size, dtype=jnp.uint32)))
    low_code = jnp.minimum(first_code, second_code)
    high_code = jnp.maximum(first_code, second_code)
    low_one_hot = jax.nn.one_hot(low_code, 256, dtype=jnp.float32)
    high_one_hot = jax.nn.one_hot(high_code, 256, dtype=jnp.float32)
    return jnp.concatenate(
        (
            pair_frequency,
            stack_frequency,
            pair_fractions,
            jnp.asarray([bounded_length], dtype=jnp.float32),
            jnp.asarray([entropy(first), entropy(second)], dtype=jnp.float32),
            low_one_hot,
            high_one_hot,
        )
    )


_motif_features_jax = jax.jit(_motif_features_kernel)
_motif_features_batch_jax = jax.jit(
    jax.vmap(_motif_features_kernel, in_axes=(0, 0, 0, 0, 0))
)


def motif_feature_vectors(
    bases_a: Any,
    bases_b: Any,
    sites_and_motifs: Sequence[tuple[BindingSite, ReactionMotif]],
) -> list[np.ndarray]:
    """Build many site feature rows in a small number of padded JAX calls."""

    first = validate_bases(bases_a)
    second = validate_bases(bases_b)
    if not sites_and_motifs:
        return []

    groups: dict[tuple[int, int], list[tuple[int, BindingSite, ReactionMotif]]] = {}
    for index, (site, motif) in enumerate(sites_and_motifs):
        first_segment = first[site.start_a : site.end_a]
        second_segment = second[site.start_b : site.end_b][::-1]
        if first_segment.size != site.length or second_segment.size != site.length:
            raise ValueError("binding site slices do not match the declared length")
        bucket_key = (
            max(_DEFAULT_KERNEL_LENGTH_CAPACITY, _bucket_size(int(site.length))),
            int(motif.motif_length),
        )
        groups.setdefault(bucket_key, []).append((index, site, motif))

    results: list[np.ndarray | None] = [None] * len(sites_and_motifs)
    for (length_bucket, motif_length), entries in groups.items():
        batch_capacity = max(_DEFAULT_KERNEL_BATCH_CAPACITY, _bucket_size(len(entries)))
        segments_a = np.zeros((batch_capacity, length_bucket), dtype=np.uint8)
        segments_b = np.zeros((batch_capacity, length_bucket), dtype=np.uint8)
        motifs_a = np.zeros((batch_capacity, motif_length), dtype=np.uint8)
        motifs_b = np.zeros((batch_capacity, motif_length), dtype=np.uint8)
        valid_lengths = np.zeros(batch_capacity, dtype=np.int32)
        for row, (_, site, motif) in enumerate(entries):
            segments_a[row, : site.length] = first[site.start_a : site.end_a]
            segments_b[row, : site.length] = second[site.start_b : site.end_b][::-1]
            motifs_a[row] = np.asarray(motif.bases_a, dtype=np.uint8)
            motifs_b[row] = np.asarray(motif.bases_b, dtype=np.uint8)
            valid_lengths[row] = site.length
        values = np.asarray(
            _motif_features_batch_jax(
                jnp.asarray(segments_a),
                jnp.asarray(segments_b),
                jnp.asarray(motifs_a),
                jnp.asarray(motifs_b),
                jnp.asarray(valid_lengths),
            ),
            dtype=np.float32,
        )
        for row, (index, _, _) in enumerate(entries):
            results[index] = values[row]

    completed = [value for value in results if value is not None]
    if len(completed) != len(results):
        raise RuntimeError("motif feature batching lost a site")
    return completed


def motif_feature_vector(
    bases_a: Any,
    bases_b: Any,
    site: BindingSite,
    motif: ReactionMotif,
) -> np.ndarray:
    """Return the fixed ``MOTIF_FEATURE_DIM`` feature vector for one site."""

    result = motif_feature_vectors(bases_a, bases_b, ((site, motif),))[0]
    if result.shape != (MOTIF_FEATURE_DIM,):
        raise RuntimeError("motif feature kernel returned an unexpected feature dimension")
    return result


def motif_statistics(motif: ReactionMotif) -> dict[str, int | float]:
    """Return compact audit statistics for one reaction motif."""

    combined = np.asarray((*motif.bases_a, *motif.bases_b), dtype=np.uint8)
    return {
        "motif_length": motif.motif_length,
        "motif_code_a": motif.code_a,
        "motif_code_b": motif.code_b,
        "motif_entropy_a": sequence_entropy(np.asarray(motif.bases_a, dtype=np.uint8)),
        "motif_entropy_b": sequence_entropy(np.asarray(motif.bases_b, dtype=np.uint8)),
        "motif_entropy_combined": sequence_entropy(combined),
    }
