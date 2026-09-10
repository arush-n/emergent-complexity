"""Seed-and-extend complementary binding for four-symbol sequences."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .alphabet import kmer_code, kmer_codes, pair_kind, reverse_complement, validate_bases

# These are deliberately dimensionless energies.  They encode the ordering
# GC/CG stronger than AU/UA stronger than GU/UG, not a thermodynamic unit.
PAIR_ENERGY: dict[tuple[int, int], float] = {
    (2, 1): -3.0,
    (1, 2): -3.0,
    (0, 3): -2.0,
    (3, 0): -2.0,
    (2, 3): -1.0,
    (3, 2): -1.0,
}


@dataclass(frozen=True)
class KmerIndex:
    """Cached contiguous k-mers for one sequence orientation."""

    bases: np.ndarray
    seed_length: int
    reverse_complement_orientation: bool
    codes: np.ndarray
    positions_by_code: dict[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        values = validate_bases(self.bases)
        if values.tobytes() != np.asarray(self.bases, dtype=np.uint8).tobytes():
            raise ValueError("k-mer index bases must be canonical uint8 values")
        if not isinstance(self.seed_length, int) or self.seed_length < 1:
            raise ValueError("seed_length must be positive")
        codes = np.asarray(self.codes, dtype=np.int64)
        if codes.ndim != 1:
            raise ValueError("k-mer codes must be one-dimensional")
        if codes.size != max(0, values.size - self.seed_length + 1):
            raise ValueError("k-mer code count does not match sequence length")
        values = values.copy()
        codes = codes.copy()
        values.setflags(write=False)
        codes.setflags(write=False)
        object.__setattr__(self, "bases", values)
        object.__setattr__(self, "codes", codes)
        object.__setattr__(
            self,
            "positions_by_code",
            {int(code): tuple(positions) for code, positions in self.positions_by_code.items()},
        )


def build_kmer_index(
    values: Any,
    seed_length: int,
    *,
    reverse_complement_orientation: bool = False,
) -> KmerIndex:
    """Build one reusable k-mer index for a species sequence."""

    bases = validate_bases(values)
    oriented = reverse_complement(bases) if reverse_complement_orientation else bases
    codes = kmer_codes(oriented, seed_length)
    positions: dict[int, list[int]] = {}
    for position, code in enumerate(codes.tolist()):
        positions.setdefault(int(code), []).append(position)
    return KmerIndex(
        oriented,
        seed_length,
        reverse_complement_orientation,
        codes,
        {code: tuple(items) for code, items in positions.items()},
    )


@jax.jit(static_argnums=(6,))
def _score_alignment_batch(
    segments_a: jax.Array,
    segments_b: jax.Array,
    valid: jax.Array,
    accessibility_a: jax.Array,
    accessibility_b: jax.Array,
    stacking_bonus: float,
    allow_gu_wobble: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Score padded candidate alignments in one accelerator kernel."""

    first = jnp.asarray(segments_a, dtype=jnp.uint8)
    second = jnp.asarray(segments_b, dtype=jnp.uint8)
    mask = jnp.asarray(valid, dtype=jnp.bool_)
    first_access = jnp.asarray(accessibility_a, dtype=jnp.float32)
    second_access = jnp.asarray(accessibility_b, dtype=jnp.float32)
    gc = ((first == 2) & (second == 1)) | ((first == 1) & (second == 2))
    au = ((first == 0) & (second == 3)) | ((first == 3) & (second == 0))
    wobble = ((first == 2) & (second == 3)) | ((first == 3) & (second == 2))
    wobble &= bool(allow_gu_wobble)
    canonical = gc | au
    mismatch = mask & ~(canonical | wobble)
    energies = jnp.where(gc, -3.0, jnp.where(au, -2.0, jnp.where(wobble, -1.0, 1.0)))
    energies = jnp.where(mask, energies, 0.0)
    pair_score = jnp.sum(energies, axis=1)
    stackable = mask[:, 1:] & mask[:, :-1] & ~mismatch[:, 1:] & ~mismatch[:, :-1]
    stack_score = jnp.sum(stackable, axis=1) * jnp.asarray(stacking_bonus, dtype=jnp.float32)
    lengths = jnp.maximum(jnp.sum(mask, axis=1), 1)
    mean_a = jnp.sum(first_access * mask, axis=1) / lengths
    mean_b = jnp.sum(second_access * mask, axis=1) / lengths
    total_score = (pair_score + stack_score) * jnp.sqrt(mean_a * mean_b)
    return (
        jnp.sum(canonical & mask, axis=1),
        jnp.sum(wobble & mask, axis=1),
        jnp.sum(mismatch, axis=1),
        pair_score,
        stack_score,
        total_score,
    )


@dataclass(frozen=True)
class BindingSite:
    """One antiparallel sequence alignment supported by a seed."""

    start_a: int
    end_a: int
    start_b: int
    end_b: int
    length: int
    canonical_pairs: int
    wobble_pairs: int
    mismatches: int
    pair_score: float
    stack_score: float
    total_score: float

    def __post_init__(self) -> None:
        for name in ("start_a", "end_a", "start_b", "end_b", "length"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end_a <= self.start_a or self.end_b <= self.start_b:
            raise ValueError("binding intervals must be non-empty")
        if self.end_a - self.start_a != self.length or self.end_b - self.start_b != self.length:
            raise ValueError("binding intervals must have the declared length")
        for name in ("canonical_pairs", "wobble_pairs", "mismatches"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.canonical_pairs + self.wobble_pairs + self.mismatches != self.length:
            raise ValueError("pair classifications must sum to binding length")
        for name in ("pair_score", "stack_score", "total_score"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class BindingResult:
    """Cached-friendly result of one seed-and-extend scan."""

    sites: tuple[BindingSite, ...]
    candidate_seed_count: int
    raw_site_count: int

    def __post_init__(self) -> None:
        if self.candidate_seed_count < 0 or self.raw_site_count < 0:
            raise ValueError("binding counts must be non-negative")


def _accessibility(values: Any, length: int) -> np.ndarray:
    if values is None:
        return np.ones(length, dtype=np.float64)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError("accessibility must be a finite vector matching sequence length")
    if np.any(result < 0.0):
        raise ValueError("accessibility values must be non-negative")
    return result


def _pair_energy(first: int, second: int, *, allow_gu_wobble: bool) -> float:
    kind = pair_kind(first, second, allow_gu_wobble=allow_gu_wobble)
    if kind == "mismatch":
        return 1.0
    return PAIR_ENERGY[(int(first), int(second))]


def _score_alignment(
    bases_a: np.ndarray,
    bases_b: np.ndarray,
    start_a: int,
    start_b_oriented: int,
    length: int,
    *,
    allow_gu_wobble: bool,
    stacking_bonus: float,
    accessibility_a: np.ndarray,
    accessibility_b: np.ndarray,
) -> BindingSite:
    """Score an alignment against B's reverse-complement orientation."""

    length_b = bases_b.size
    canonical = 0
    wobble = 0
    mismatches = 0
    pair_score = 0.0
    stack_score = 0.0
    paired_a_accessibility: list[float] = []
    paired_b_accessibility: list[float] = []
    previous_kind = "mismatch"
    for offset in range(length):
        position_a = start_a + offset
        oriented_position_b = start_b_oriented + offset
        position_b = length_b - 1 - oriented_position_b
        first = int(bases_a[position_a])
        second = int(bases_b[position_b])
        kind = pair_kind(first, second, allow_gu_wobble=allow_gu_wobble)
        if kind == "canonical":
            canonical += 1
        elif kind == "wobble":
            wobble += 1
        else:
            mismatches += 1
        pair_score += _pair_energy(first, second, allow_gu_wobble=allow_gu_wobble)
        if offset and previous_kind != "mismatch" and kind != "mismatch":
            stack_score += float(stacking_bonus)
        previous_kind = kind
        paired_a_accessibility.append(float(accessibility_a[position_a]))
        paired_b_accessibility.append(float(accessibility_b[position_b]))

    accessibility_factor = float(
        np.sqrt(np.mean(paired_a_accessibility) * np.mean(paired_b_accessibility))
    )
    unadjusted_score = pair_score + stack_score
    total_score = unadjusted_score * accessibility_factor
    return BindingSite(
        start_a=start_a,
        end_a=start_a + length,
        start_b=length_b - (start_b_oriented + length),
        end_b=length_b - start_b_oriented,
        length=length,
        canonical_pairs=canonical,
        wobble_pairs=wobble,
        mismatches=mismatches,
        pair_score=float(pair_score),
        stack_score=float(stack_score),
        total_score=float(total_score),
    )


def _extend_seed(
    bases_a: np.ndarray,
    bases_b: np.ndarray,
    start_a: int,
    start_b_oriented: int,
    seed_length: int,
    *,
    allow_gu_wobble: bool,
    max_mismatches: int,
) -> tuple[int, int, int, int]:
    """Extend a canonical seed while allowing a bounded mismatch budget."""

    length_b = bases_b.size
    left_a = start_a
    left_b = start_b_oriented
    right_a = start_a + seed_length
    right_b = start_b_oriented + seed_length
    mismatches = 0

    while left_a > 0 and left_b > 0:
        position_b = length_b - left_b
        kind = pair_kind(
            int(bases_a[left_a - 1]),
            int(bases_b[position_b]),
            allow_gu_wobble=allow_gu_wobble,
        )
        if kind == "mismatch":
            if mismatches >= max_mismatches:
                break
            mismatches += 1
        left_a -= 1
        left_b -= 1

    while right_a < bases_a.size and right_b < bases_b.size:
        position_b = length_b - 1 - right_b
        kind = pair_kind(
            int(bases_a[right_a]),
            int(bases_b[position_b]),
            allow_gu_wobble=allow_gu_wobble,
        )
        if kind == "mismatch":
            if mismatches >= max_mismatches:
                break
            mismatches += 1
        right_a += 1
        right_b += 1
    return left_a, right_a, left_b, right_b


def _overlap(first_start: int, first_end: int, second_start: int, second_end: int) -> bool:
    return first_start < second_end and second_start < first_end


def _compatible_seed_codes(seed: np.ndarray, *, allow_gu_wobble: bool) -> tuple[int, ...]:
    """Return oriented-B seed codes compatible with one A seed."""

    choices: list[tuple[int, ...]] = []
    for value in seed.tolist():
        options = [int(value)]
        if allow_gu_wobble and int(value) == 2:
            # G in A paired with U in original B appears as A when oriented.
            options.append(0)
        elif allow_gu_wobble and int(value) == 3:
            # U in A paired with G in original B appears as C when oriented.
            options.append(1)
        choices.append(tuple(options))
    return tuple(
        sorted(
            {kmer_code(np.asarray(candidate, dtype=np.uint8)) for candidate in product(*choices)}
        )
    )


def _select_non_overlapping(sites: list[BindingSite]) -> list[BindingSite]:
    """Keep strongest sites while preventing reuse of sequence intervals."""

    ordered = sorted(
        sites,
        key=lambda site: (
            float(site.total_score),
            -site.length,
            site.start_a,
            site.start_b,
            site.end_a,
            site.end_b,
        ),
    )
    selected: list[BindingSite] = []
    for candidate in ordered:
        if any(
            _overlap(candidate.start_a, candidate.end_a, other.start_a, other.end_a)
            or _overlap(candidate.start_b, candidate.end_b, other.start_b, other.end_b)
            for other in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda site: (site.start_a, site.start_b, site.end_a, site.end_b))


def scan_binding_sites(
    bases_a: Any,
    bases_b: Any,
    *,
    seed_length: int = 4,
    minimum_length: int = 4,
    allow_gu_wobble: bool = True,
    max_mismatches: int = 1,
    stacking_bonus: float = -0.5,
    accessibility_a: Any = None,
    accessibility_b: Any = None,
    kmer_index_a: KmerIndex | None = None,
    kmer_index_b: KmerIndex | None = None,
) -> BindingResult:
    """Find complementary antiparallel sites with deterministic seed-and-extend."""

    first = validate_bases(bases_a)
    second = validate_bases(bases_b)
    if not isinstance(seed_length, int) or seed_length < 1:
        raise ValueError("seed_length must be a positive integer")
    if not isinstance(minimum_length, int) or minimum_length < seed_length:
        raise ValueError("minimum_length must be at least seed_length")
    if not isinstance(max_mismatches, int) or max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative")
    if not isinstance(allow_gu_wobble, bool):
        raise TypeError("allow_gu_wobble must be a boolean")
    if not isinstance(stacking_bonus, (int, float)) or not np.isfinite(stacking_bonus):
        raise ValueError("stacking_bonus must be finite")
    first_accessibility = _accessibility(accessibility_a, first.size)
    second_accessibility = _accessibility(accessibility_b, second.size)

    first_index = kmer_index_a or build_kmer_index(first, seed_length)
    second_index = kmer_index_b or build_kmer_index(
        second,
        seed_length,
        reverse_complement_orientation=True,
    )
    if first_index.seed_length != seed_length or second_index.seed_length != seed_length:
        raise ValueError("k-mer index seed lengths must match the scan")
    if first_index.bases.tobytes() != first.tobytes():
        raise ValueError("kmer_index_a does not match bases_a")
    if second_index.bases.tobytes() != reverse_complement(second).tobytes():
        raise ValueError("kmer_index_b does not match bases_b's reverse complement")
    first_codes = first_index.codes
    second_codes = second_index.codes
    if first_codes.size == 0 or second_codes.size == 0:
        return BindingResult((), 0, 0)

    candidates: set[tuple[int, int]] = set()
    for start_a, _ in enumerate(first_codes.tolist()):
        seed = first[start_a : start_a + seed_length]
        for code in _compatible_seed_codes(seed, allow_gu_wobble=allow_gu_wobble):
            for start_b_oriented in second_index.positions_by_code.get(code, ()):
                candidates.add((start_a, int(start_b_oriented)))

    extents: list[tuple[int, int, int, int]] = []
    for start_a, start_b_oriented in sorted(candidates):
        left_a, right_a, left_b, right_b = _extend_seed(
            first,
            second,
            start_a,
            start_b_oriented,
            seed_length,
            allow_gu_wobble=allow_gu_wobble,
            max_mismatches=max_mismatches,
        )
        length = right_a - left_a
        if length < minimum_length:
            continue
        extents.append((left_a, right_a, left_b, right_b))
    if not extents:
        return BindingResult((), len(candidates), 0)

    maximum_length = max(right_a - left_a for left_a, right_a, _, _ in extents)
    segments_a = np.zeros((len(extents), maximum_length), dtype=np.uint8)
    segments_b = np.zeros((len(extents), maximum_length), dtype=np.uint8)
    valid = np.zeros((len(extents), maximum_length), dtype=bool)
    segment_accessibility_a = np.ones((len(extents), maximum_length), dtype=np.float32)
    segment_accessibility_b = np.ones((len(extents), maximum_length), dtype=np.float32)
    for index, (left_a, right_a, left_b, right_b) in enumerate(extents):
        length = right_a - left_a
        segments_a[index, :length] = first[left_a:right_a]
        segments_b[index, :length] = second[second.size - right_b : second.size - left_b][::-1]
        valid[index, :length] = True
        segment_accessibility_a[index, :length] = first_accessibility[left_a:right_a]
        segment_accessibility_b[index, :length] = second_accessibility[
            second.size - right_b : second.size - left_b
        ][::-1]
    canonical_counts, wobble_counts, mismatch_counts, pair_scores, stack_scores, total_scores = (
        _score_alignment_batch(
            segments_a,
            segments_b,
            valid,
            segment_accessibility_a,
            segment_accessibility_b,
            float(stacking_bonus),
            allow_gu_wobble,
        )
    )
    raw: dict[tuple[int, int, int, int], BindingSite] = {}
    for index, (left_a, right_a, left_b, right_b) in enumerate(extents):
        length = right_a - left_a
        site = BindingSite(
            start_a=left_a,
            end_a=right_a,
            start_b=second.size - right_b,
            end_b=second.size - left_b,
            length=length,
            canonical_pairs=int(canonical_counts[index]),
            wobble_pairs=int(wobble_counts[index]),
            mismatches=int(mismatch_counts[index]),
            pair_score=float(pair_scores[index]),
            stack_score=float(stack_scores[index]),
            total_score=float(total_scores[index]),
        )
        raw[(site.start_a, site.end_a, site.start_b, site.end_b)] = site
    selected = _select_non_overlapping(list(raw.values()))
    return BindingResult(tuple(selected), len(candidates), len(raw))


def find_binding_sites(
    bases_a: Any,
    bases_b: Any,
    **kwargs: Any,
) -> list[BindingSite]:
    """Convenience wrapper returning only the selected non-overlapping sites."""

    return list(scan_binding_sites(bases_a, bases_b, **kwargs).sites)
