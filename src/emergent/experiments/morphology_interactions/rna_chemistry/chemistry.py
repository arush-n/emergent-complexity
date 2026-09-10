"""JAX-accelerated motif chemistry and deterministic pair caches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..canonical import ShapeKey, shape_key_sort_key
from ..encoding import deterministic_uniform
from .accessibility import AccessibilityCache
from .binding import BindingResult, BindingSite, KmerIndex, scan_binding_sites
from .motifs import MOTIF_FEATURE_DIM, ReactionMotif, extract_reaction_motif, motif_feature_vector
from .sequence import ChemicalSequence

CHANNEL_COUNT = 18
UINT64_SCALE = float(2**64)


@jax.jit
def _structured_batch_kernel(
    features: jax.Array,
    weights: jax.Array,
    gain: float,
    beta: float,
) -> jax.Array:
    """Evaluate a batch of motif features in one compiled matrix operation."""

    values = jnp.asarray(features, dtype=jnp.float32)
    matrices = jnp.asarray(weights, dtype=jnp.float32)
    dimension = values.shape[-1]
    raw = values @ jnp.swapaxes(matrices, 0, 1) / jnp.sqrt(float(dimension))
    return jnp.tanh(
        jnp.asarray(beta, dtype=jnp.float32) * jnp.asarray(gain, dtype=jnp.float32) * raw
    )


@jax.jit
def _interpolate_kernel(
    structured: jax.Array,
    scrambled: jax.Array,
    alpha: float,
) -> jax.Array:
    """Mix calibrated site vectors on the active JAX backend."""

    alpha_value = jnp.asarray(alpha, dtype=jnp.float32)
    return (1.0 - alpha_value) * structured + alpha_value * scrambled


@dataclass(frozen=True)
class ChemistryCalibration:
    """Fixed corpus calibration recorded with every chemistry universe."""

    corpus_size: int
    structured_gain: float
    target_rms: float
    structured_rms: float
    scrambled_rms: float
    structured_threshold_rate: float
    scrambled_threshold_rate: float
    threshold: float

    def as_dict(self) -> dict[str, int | float]:
        """Return JSON-compatible calibration measurements."""

        return {
            "corpus_size": self.corpus_size,
            "structured_gain": self.structured_gain,
            "target_rms": self.target_rms,
            "structured_rms": self.structured_rms,
            "scrambled_rms": self.scrambled_rms,
            "structured_threshold_rate": self.structured_threshold_rate,
            "scrambled_threshold_rate": self.scrambled_threshold_rate,
            "threshold": self.threshold,
        }


def _make_weights(seed: int, feature_dim: int) -> np.ndarray:
    """Generate the one fixed motif projection for a universe."""

    values = deterministic_uniform(
        seed,
        "rna-motif-projection",
        CHANNEL_COUNT * feature_dim,
        low=-1.0,
        high=1.0,
    ).reshape((CHANNEL_COUNT, feature_dim))
    # Per-channel normalization makes the calibration meaningful and avoids a
    # channel receiving a larger action budget merely from its raw projection.
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    values.setflags(write=False)
    return values.astype(np.float32)


def _calibrate(
    seed: int,
    weights: np.ndarray,
    *,
    threshold: float,
    corpus_size: int,
    beta: float,
) -> ChemistryCalibration:
    """Calibrate structured magnitudes against a fixed deterministic corpus."""

    feature_dim = weights.shape[1]
    raw_features = deterministic_uniform(
        seed,
        "rna-calibration-features",
        corpus_size * feature_dim,
        low=-1.0,
        high=1.0,
    ).reshape((corpus_size, feature_dim))
    raw_features /= np.maximum(np.linalg.norm(raw_features, axis=1, keepdims=True), 1e-12)
    scrambled = deterministic_uniform(
        seed,
        "rna-calibration-scrambled",
        corpus_size * CHANNEL_COUNT,
        low=-1.0,
        high=1.0,
    ).reshape((corpus_size, CHANNEL_COUNT))
    target_rms = float(np.sqrt(np.mean(scrambled**2)))

    def structured_for(gain: float) -> np.ndarray:
        return np.asarray(
            _structured_batch_kernel(raw_features, weights, gain, beta),
            dtype=np.float32,
        )

    low = 0.0
    high = 1.0
    while float(np.sqrt(np.mean(structured_for(high) ** 2))) < target_rms and high < 1_000_000.0:
        high *= 2.0
    for _ in range(36):
        middle = 0.5 * (low + high)
        if float(np.sqrt(np.mean(structured_for(middle) ** 2))) < target_rms:
            low = middle
        else:
            high = middle
    gain = 0.5 * (low + high)
    structured = structured_for(gain)
    return ChemistryCalibration(
        corpus_size=corpus_size,
        structured_gain=float(gain),
        target_rms=target_rms,
        structured_rms=float(np.sqrt(np.mean(structured**2))),
        scrambled_rms=float(np.sqrt(np.mean(scrambled**2))),
        structured_threshold_rate=float(np.mean(np.abs(structured) >= threshold)),
        scrambled_threshold_rate=float(np.mean(np.abs(scrambled) >= threshold)),
        threshold=float(threshold),
    )


@dataclass(frozen=True)
class ChemistryUniverse:
    """Fixed projection, calibration, and alpha law for one run."""

    universe_seed: int
    beta: float = 1.0
    calibration_threshold: float = 0.35
    calibration_size: int = 512
    feature_dim: int = MOTIF_FEATURE_DIM
    weights: np.ndarray = field(init=False, repr=False)
    calibration: ChemistryCalibration = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.universe_seed, int):
            raise TypeError("universe_seed must be an integer")
        if not isinstance(self.beta, (int, float)) or float(self.beta) < 0.0:
            raise ValueError("beta must be a non-negative number")
        if not isinstance(self.calibration_size, int) or self.calibration_size < 32:
            raise ValueError("calibration_size must be at least 32")
        if not isinstance(self.feature_dim, int) or self.feature_dim != MOTIF_FEATURE_DIM:
            raise ValueError(f"feature_dim must equal MOTIF_FEATURE_DIM={MOTIF_FEATURE_DIM}")
        if not 0.0 <= float(self.calibration_threshold) <= 1.0:
            raise ValueError("calibration_threshold must be between 0 and 1")
        weights = _make_weights(self.universe_seed, self.feature_dim)
        calibration = _calibrate(
            self.universe_seed,
            weights,
            threshold=float(self.calibration_threshold),
            corpus_size=self.calibration_size,
            beta=float(self.beta),
        )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "calibration", calibration)

    @property
    def seed(self) -> int:
        """Short manifest-friendly seed alias."""

        return self.universe_seed

    def structured_vector(self, features: Any) -> np.ndarray:
        """Evaluate one calibrated structured motif vector through JAX."""

        values = np.asarray(features, dtype=np.float32)
        if values.shape != (self.feature_dim,):
            raise ValueError(f"features must have shape ({self.feature_dim},)")
        result = np.asarray(
            _structured_batch_kernel(
                values[None, :],
                self.weights,
                self.calibration.structured_gain,
                float(self.beta),
            )[0],
            dtype=np.float32,
        )
        result.setflags(write=False)
        return result

    def structured_batch(self, features: Any) -> np.ndarray:
        """Evaluate many motif feature rows in one JAX call."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (batch, {self.feature_dim})")
        return np.asarray(
            _structured_batch_kernel(
                values,
                self.weights,
                self.calibration.structured_gain,
                float(self.beta),
            ),
            dtype=np.float32,
        )

    def scrambled_vector(self, motif: ReactionMotif) -> np.ndarray:
        """Return a symmetric BLAKE2b control vector for one exact motif pair."""

        low_code, high_code = motif.symmetric_codes
        values = []
        for channel in range(CHANNEL_COUNT):
            digest = hashlib.blake2b(digest_size=8, person=b"rna-scramble")
            digest.update((self.universe_seed & ((1 << 64) - 1)).to_bytes(8, "little"))
            digest.update(int(motif.motif_length).to_bytes(4, "little"))
            digest.update(int(low_code).to_bytes(8, "little"))
            digest.update(int(high_code).to_bytes(8, "little"))
            digest.update(int(channel).to_bytes(4, "little"))
            word = int.from_bytes(digest.digest(), "little")
            values.append(2.0 * (word / UINT64_SCALE) - 1.0)
        result = np.asarray(values, dtype=np.float32)
        result.setflags(write=False)
        return result

    def mix_vectors(self, structured: Any, scrambled: Any, *, alpha: float) -> np.ndarray:
        """Interpolate two calibrated vectors through one compiled kernel."""

        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        first = np.asarray(structured, dtype=np.float32)
        second = np.asarray(scrambled, dtype=np.float32)
        if first.shape != (CHANNEL_COUNT,) or second.shape != (CHANNEL_COUNT,):
            raise ValueError("site vectors must contain exactly 18 channels")
        result = np.asarray(_interpolate_kernel(first, second, float(alpha)), dtype=np.float32)
        result.setflags(write=False)
        return result

    def mix_batch(self, structured: Any, scrambled: Any, *, alpha: float) -> np.ndarray:
        """Mix many calibrated vectors in one compiled JAX call."""

        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        first = np.asarray(structured, dtype=np.float32)
        second = np.asarray(scrambled, dtype=np.float32)
        if first.ndim != 2 or first.shape[1] != CHANNEL_COUNT or second.shape != first.shape:
            raise ValueError("batched site vectors must have shape (batch, 18)")
        return np.asarray(_interpolate_kernel(first, second, float(alpha)), dtype=np.float32)

    def calibration_report(self) -> dict[str, int | float]:
        """Return the fixed-universe magnitude calibration."""

        return self.calibration.as_dict()


def make_chemistry_universe(
    universe_seed: int,
    *,
    beta: float = 1.0,
    calibration_threshold: float = 0.35,
    calibration_size: int = 512,
) -> ChemistryUniverse:
    """Construct one deterministic chemistry universe."""

    return ChemistryUniverse(
        universe_seed,
        beta=beta,
        calibration_threshold=calibration_threshold,
        calibration_size=calibration_size,
    )


@dataclass(frozen=True)
class SiteInteraction:
    """One successful binding site's deterministic reaction output."""

    site: BindingSite
    motif: ReactionMotif
    feature_vector: np.ndarray
    structured_vector: np.ndarray
    scrambled_vector: np.ndarray
    final_vector: np.ndarray
    binding_score: float
    lifetime: int

    def __post_init__(self) -> None:
        for name in (
            "feature_vector",
            "structured_vector",
            "scrambled_vector",
            "final_vector",
        ):
            values = np.asarray(getattr(self, name))
            if name == "feature_vector" and values.shape != (MOTIF_FEATURE_DIM,):
                raise ValueError("feature_vector has the wrong dimension")
            if name != "feature_vector" and values.shape != (CHANNEL_COUNT,):
                raise ValueError("reaction vectors must contain 18 channels")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        if not np.all((-1.0 <= self.final_vector) & (self.final_vector <= 1.0)):
            raise ValueError("final_vector must lie in [-1, 1]")
        if self.lifetime < 1:
            raise ValueError("lifetime must be positive")
        for name in (
            "feature_vector",
            "structured_vector",
            "scrambled_vector",
            "final_vector",
        ):
            value = np.asarray(getattr(self, name))
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def strength(self) -> float:
        """Return deterministic reaction strength used for overlap ordering."""

        return float(np.mean(np.abs(self.final_vector)))


@dataclass
class PairChemistry:
    """Cached chemistry for one unordered species pair."""

    species_a: ShapeKey
    species_b: ShapeKey
    candidate_seed_count: int
    raw_site_count: int
    sites: tuple[SiteInteraction, ...]
    best_binding_score: float
    total_paired_bases: int
    first_seen_generation: int = -1
    total_site_effect_area: int = 0
    encounters: int = 0

    def __post_init__(self) -> None:
        first, second = sorted((self.species_a, self.species_b), key=shape_key_sort_key)
        self.species_a = first
        self.species_b = second

    @property
    def pair_key(self) -> tuple[ShapeKey, ShapeKey]:
        """Return the stable unordered cache key."""

        return self.species_a, self.species_b

    @property
    def successful_binding_sites(self) -> int:
        """Return the number of selected energetically viable sites."""

        return len(self.sites)

    @property
    def distinct_site_rules(self) -> int:
        """Return the number of distinct continuous site vectors."""

        return len({site.final_vector.tobytes() for site in self.sites})


def _site_lifetime(
    energy: float,
    *,
    mode: str,
    max_lifetime: int,
    energy_offset: float,
    temperature: float,
) -> int:
    if mode == "instant":
        return 1
    if mode != "energy":
        raise ValueError("binding lifetime mode must be instant or energy")
    scaled = (-float(energy) - float(energy_offset)) / float(temperature)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))
    return 1 + int(np.floor(max_lifetime * sigmoid))


def evaluate_pair_chemistry(
    sequence_a: ChemicalSequence,
    sequence_b: ChemicalSequence,
    *,
    universe: ChemistryUniverse,
    alpha: float = 0.0,
    seed_length: int = 4,
    minimum_length: int = 4,
    allow_gu_wobble: bool = True,
    max_mismatches: int = 1,
    stacking_bonus: float = -0.5,
    binding_energy_threshold: float = -5.0,
    motif_length: int = 4,
    accessibility_a: Any = None,
    accessibility_b: Any = None,
    kmer_index_a: KmerIndex | None = None,
    kmer_index_b: KmerIndex | None = None,
    binding_lifetime_mode: str = "instant",
    max_binding_lifetime: int = 16,
    lifetime_energy_offset: float = 5.0,
    lifetime_temperature: float = 2.0,
) -> PairChemistry:
    """Separate binding recognition from deterministic site reaction chemistry."""

    if not isinstance(sequence_a, ChemicalSequence) or not isinstance(sequence_b, ChemicalSequence):
        raise TypeError("sequence_a and sequence_b must be ChemicalSequence values")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if float(lifetime_temperature) <= 0.0:
        raise ValueError("lifetime_temperature must be positive")

    if shape_key_sort_key(sequence_a.shape_key) <= shape_key_sort_key(sequence_b.shape_key):
        first, second = sequence_a, sequence_b
        first_accessibility, second_accessibility = accessibility_a, accessibility_b
        first_index, second_index = kmer_index_a, kmer_index_b
    else:
        first, second = sequence_b, sequence_a
        first_accessibility, second_accessibility = accessibility_b, accessibility_a
        first_index, second_index = kmer_index_b, kmer_index_a
    scan: BindingResult = scan_binding_sites(
        first.bases,
        second.bases,
        seed_length=seed_length,
        minimum_length=minimum_length,
        allow_gu_wobble=allow_gu_wobble,
        max_mismatches=max_mismatches,
        stacking_bonus=stacking_bonus,
        accessibility_a=first_accessibility,
        accessibility_b=second_accessibility,
        kmer_index_a=first_index,
        kmer_index_b=second_index,
    )
    site_records: list[tuple[BindingSite, ReactionMotif, np.ndarray]] = []
    for site in scan.sites:
        if site.total_score > float(binding_energy_threshold):
            continue
        motif = extract_reaction_motif(
            first.bases,
            second.bases,
            site,
            motif_length=motif_length,
            accessibility_a=first_accessibility,
            accessibility_b=second_accessibility,
        )
        features = motif_feature_vector(first.bases, second.bases, site, motif)
        site_records.append((site, motif, features))
    interactions: list[SiteInteraction] = []
    if site_records:
        structured_vectors = universe.structured_batch(
            np.asarray([record[2] for record in site_records], dtype=np.float32)
        )
        scrambled_vectors = np.asarray(
            [universe.scrambled_vector(record[1]) for record in site_records],
            dtype=np.float32,
        )
        final_vectors = universe.mix_batch(structured_vectors, scrambled_vectors, alpha=alpha)
    else:
        structured_vectors = np.zeros((0, CHANNEL_COUNT), dtype=np.float32)
        scrambled_vectors = np.zeros((0, CHANNEL_COUNT), dtype=np.float32)
        final_vectors = np.zeros((0, CHANNEL_COUNT), dtype=np.float32)
    for index, (site, motif, features) in enumerate(site_records):
        interactions.append(
            SiteInteraction(
                site=site,
                motif=motif,
                feature_vector=features,
                structured_vector=structured_vectors[index],
                scrambled_vector=scrambled_vectors[index],
                final_vector=final_vectors[index],
                binding_score=float(site.total_score),
                lifetime=_site_lifetime(
                    site.total_score,
                    mode=binding_lifetime_mode,
                    max_lifetime=max_binding_lifetime,
                    energy_offset=lifetime_energy_offset,
                    temperature=lifetime_temperature,
                ),
            )
        )
    return PairChemistry(
        species_a=first.shape_key,
        species_b=second.shape_key,
        candidate_seed_count=scan.candidate_seed_count,
        raw_site_count=scan.raw_site_count,
        sites=tuple(interactions),
        best_binding_score=(
            float(min(site.binding_score for site in interactions)) if interactions else 0.0
        ),
        total_paired_bases=sum(site.site.length for site in interactions),
    )


def cached_pair_chemistry(
    cache: dict[tuple[ShapeKey, ShapeKey], PairChemistry],
    sequence_a: ChemicalSequence,
    sequence_b: ChemicalSequence,
    *,
    encounter_generation: int | None = None,
    **kwargs: Any,
) -> PairChemistry:
    """Look up or calculate one pair; all expensive chemistry is cached."""

    key = tuple(sorted((sequence_a.shape_key, sequence_b.shape_key), key=shape_key_sort_key))
    result = cache.get(key)
    if result is None:
        result = evaluate_pair_chemistry(sequence_a, sequence_b, **kwargs)
        if encounter_generation is not None:
            result.first_seen_generation = int(encounter_generation)
        cache[key] = result
    elif result.first_seen_generation < 0 and encounter_generation is not None:
        result.first_seen_generation = int(encounter_generation)
    result.encounters += 1
    return result


def accessibility_cache_for_config(
    *,
    mode: str,
    minimum_hairpin_separation: int,
    paired_accessibility: float,
    allow_gu_wobble: bool,
) -> AccessibilityCache:
    """Build the cache object used by the engine and offline audits."""

    return AccessibilityCache(
        mode=mode,
        minimum_hairpin_separation=minimum_hairpin_separation,
        paired_accessibility=paired_accessibility,
        allow_gu_wobble=allow_gu_wobble,
    )
