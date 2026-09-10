"""Standalone deterministic engine for RNA-inspired morphology chemistry."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ....core.random import key_from_seed, random_grid
from ....core.rules import Rule, format_rule, rule_to_masks
from ....core.step import step_jit
from ....io.patterns import get_pattern
from ....io.serialization import load_grid
from ..canonical import ShapeKey, canonicalize_component, matrix_from_shape_key, shape_key_sort_key
from ..components import Component, detect_components
from ..interaction import find_interacting_pairs
from ..local_step import step_with_interactions
from ..native_api import load_native_grid
from .accessibility import AccessibilityCache
from .alphabet import base_frequencies, kmer_codes, sequence_entropy
from .binding import KmerIndex, build_kmer_index
from .chemistry import (
    ChemistryUniverse,
    PairChemistry,
    cached_pair_chemistry,
    make_chemistry_universe,
)
from .config import RNAExperimentConfig
from .effects import (
    SiteEffect,
    resolve_site_owner_map,
    site_effect_zone,
    site_interaction_to_effect,
    site_rule_arrays,
)
from .metrics import GENERATION_FIELDS, ChemicalMetricsTracker
from .sequence import ChemicalSequence, morphology_perimeter, sequence_from_shape_key

ARTIFACT_ROOT = Path("artifacts/experiments/morphology_interactions/rna_chemistry")


def _host_grid(grid: Any) -> np.ndarray:
    values = np.asarray(jax.device_get(grid), dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    return (values != 0).astype(np.uint8)


def _place_pattern(
    height: int,
    width: int,
    pattern_name: str,
    row: int | None,
    col: int | None,
) -> np.ndarray:
    pattern = np.asarray(get_pattern(pattern_name), dtype=np.uint8)
    result = np.zeros((height, width), dtype=np.uint8)
    start_row = (height - pattern.shape[0]) // 2 if row is None else row
    start_col = (width - pattern.shape[1]) // 2 if col is None else col
    for pattern_row, pattern_col in np.argwhere(pattern != 0):
        result[(start_row + int(pattern_row)) % height, (start_col + int(pattern_col)) % width] = 1
    return result


def initial_grid_from_config(config: RNAExperimentConfig) -> np.ndarray:
    """Create the deterministic initial grid selected by the configuration."""

    if config.initial_condition == "random":
        return _host_grid(
            random_grid(key_from_seed(config.seed), config.height, config.width, config.density)
        )
    if config.initial_condition == "patterns":
        return _place_pattern(
            config.height,
            config.width,
            config.pattern_name,
            config.pattern_row,
            config.pattern_col,
        )
    if config.initial_condition == "npz":
        return _host_grid(load_grid(config.initial_condition_path or ""))
    if config.initial_condition == "api":
        return _host_grid(
            load_native_grid(
                config.native_base_url,
                width=config.width,
                height=config.height,
                density=config.density,
                seed=config.seed,
                rule=config.base_rule,
            )
        )
    raise ValueError(f"unsupported initial_condition: {config.initial_condition}")


@dataclass
class ChemicalSpeciesRecord:
    """Recurrence bookkeeping for one exact morphology and its sequence."""

    key: ShapeKey
    sequence: ChemicalSequence
    first_seen_generation: int
    last_seen_generation: int
    observations: int = 1
    independent_components: int = 1
    species_id: int = -1


class ChemicalSpeciesRegistry(dict[ShapeKey, ChemicalSpeciesRecord]):
    """Insertion-ordered exact-shape registry used by one engine."""

    def observe(
        self,
        key: ShapeKey,
        sequence: ChemicalSequence,
        *,
        generation: int,
    ) -> tuple[ChemicalSpeciesRecord, bool]:
        """Register one component observation."""

        record = self.get(key)
        is_new = record is None
        if record is None:
            record = ChemicalSpeciesRecord(
                key=key,
                sequence=sequence,
                first_seen_generation=generation,
                last_seen_generation=generation,
                species_id=len(self),
            )
            self[key] = record
        else:
            record.observations += 1
            record.independent_components += 1
            record.last_seen_generation = generation
        return record, is_new

    def records_by_id(self) -> list[ChemicalSpeciesRecord]:
        """Return records in stable first-seen order."""

        return sorted(self.values(), key=lambda record: record.species_id)


@dataclass(frozen=True)
class ActiveSite:
    """A cached chemical site placed in one current spatial encounter."""

    pair_key: tuple[ShapeKey, ShapeKey]
    chemistry: PairChemistry
    effect: SiteEffect
    zone: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.zone, dtype=bool).copy()
        values.setflags(write=False)
        object.__setattr__(self, "zone", values)


@dataclass(frozen=True)
class StepResult:
    """Host diagnostics for one ``X_t -> X_(t+1)`` transition."""

    source_generation: int
    generation: int
    grid: np.ndarray
    components: tuple[Component, ...]
    observations: tuple[tuple[Component, ShapeKey, ChemicalSequence], ...]
    pair_chemistries: tuple[PairChemistry, ...]
    active_sites: tuple[ActiveSite, ...]
    owner_map: np.ndarray
    detection_performed: bool
    metrics: dict[str, int | float] | None


@dataclass
class RNAExperimentResult:
    """Complete in-memory result for artifact writing and analysis."""

    config: RNAExperimentConfig
    initial_grid: np.ndarray
    final_grid: np.ndarray
    universe: ChemistryUniverse
    species_registry: ChemicalSpeciesRegistry
    pair_cache: dict[tuple[ShapeKey, ShapeKey], PairChemistry]
    motifs_by_species: dict[ShapeKey, int]
    generation_records: list[dict[str, int | float]]
    snapshots: dict[int, np.ndarray]
    summary: dict[str, int | float]


class RNAChemistryEngine:
    """Execute one deterministic RNA-inspired chemistry universe."""

    def __init__(
        self,
        config: RNAExperimentConfig,
        *,
        initial_grid: Any | None = None,
        universe: ChemistryUniverse | None = None,
    ) -> None:
        if not isinstance(config, RNAExperimentConfig):
            raise TypeError("config must be an RNAExperimentConfig")
        self.config = config
        self.base_rule: Rule = config.rule
        self.base_birth, self.base_survival = rule_to_masks(self.base_rule)
        self.universe = universe or make_chemistry_universe(
            config.seed,
            beta=config.beta,
            calibration_threshold=config.interaction_threshold,
            calibration_size=config.calibration_size,
        )
        if not isinstance(self.universe, ChemistryUniverse):
            raise TypeError("universe must be a ChemistryUniverse")
        if universe is not None and universe.universe_seed != config.seed:
            raise ValueError("universe seed must match config.seed")
        initial = (
            initial_grid_from_config(config) if initial_grid is None else _host_grid(initial_grid)
        )
        expected_shape = (config.height, config.width)
        if initial.shape != expected_shape:
            raise ValueError(f"initial grid shape must be {expected_shape}, got {initial.shape}")
        self.initial_grid = initial.copy()
        self.grid = jnp.asarray(initial, dtype=jnp.uint8)
        self.generation = 0
        self.species_registry = ChemicalSpeciesRegistry()
        self.sequence_cache: dict[ShapeKey, ChemicalSequence] = {}
        self.kmer_index_cache: dict[tuple[ShapeKey, int, bool], KmerIndex] = {}
        self.pair_cache: dict[tuple[ShapeKey, ShapeKey], PairChemistry] = {}
        self._canonical_cache: dict[tuple[tuple[int, int], bytes], ShapeKey] = {}
        self.accessibility_cache = AccessibilityCache(
            mode=config.accessibility_mode,
            minimum_hairpin_separation=config.minimum_hairpin_separation,
            paired_accessibility=config.paired_accessibility,
            allow_gu_wobble=config.allow_gu_wobble,
            max_nussinov_length=config.max_nussinov_length,
            window=config.fold_window,
        )
        self.metrics = ChemicalMetricsTracker(config.height * config.width)
        self.snapshots: dict[int, np.ndarray] = {0: initial.copy()}
        self.last_step: StepResult | None = None
        self._effect_slot_capacity = max(1, config.height * config.width)
        self._bound_sites: dict[tuple[Any, ...], tuple[ActiveSite, int]] = {}

    @property
    def height(self) -> int:
        """World height."""

        return self.config.height

    @property
    def width(self) -> int:
        """World width."""

        return self.config.width

    def _observe_current(
        self,
        current: np.ndarray,
    ) -> tuple[list[Component], list[tuple[Component, ShapeKey, ChemicalSequence]], int]:
        components = detect_components(
            current,
            min_component_cells=self.config.min_component_cells,
            backend=self.config.component_backend,
        )
        return self._observe_components(components)

    def _observe_components(
        self,
        components: list[Component],
    ) -> tuple[list[Component], list[tuple[Component, ShapeKey, ChemicalSequence]], int]:
        """Register pre-detected components without running detection twice."""

        observations: list[tuple[Component, ShapeKey, ChemicalSequence]] = []
        new_species = 0
        for component in components:
            key = self._canonical_key(component)
            sequence = self.sequence_cache.get(key)
            if sequence is None:
                sequence = sequence_from_shape_key(key, mode=self.config.sequence_mode)
                self.sequence_cache[key] = sequence
            _, is_new = self.species_registry.observe(
                key,
                sequence,
                generation=self.generation,
            )
            new_species += int(is_new)
            observations.append((component, key, sequence))
        return components, observations, new_species

    def _canonical_key(self, component: Component) -> ShapeKey:
        """Return one exact key, reusing the translation-invariant host cache."""

        shifted = component.coordinates - component.coordinates.min(axis=0)
        cache_key: tuple[tuple[int, int], bytes] | None = None
        if (
            int(shifted[:, 0].max()) < self.height - 1
            and int(shifted[:, 1].max()) < self.width - 1
        ):
            cache_key = (tuple(shifted.shape), shifted.tobytes())
        key = self._canonical_cache.get(cache_key) if cache_key is not None else None
        if key is None:
            key = canonicalize_component(
                component,
                grid_shape=(self.height, self.width),
                rotation_invariant=self.config.rotation_invariant,
                reflection_invariant=self.config.reflection_invariant,
            )
            if cache_key is not None:
                self._canonical_cache[cache_key] = key
        return key

    def _pair_chemistry(
        self,
        first: tuple[Component, ShapeKey, ChemicalSequence],
        second: tuple[Component, ShapeKey, ChemicalSequence],
    ) -> PairChemistry:
        first_key = first[1]
        second_key = second[1]
        if shape_key_sort_key(first_key) <= shape_key_sort_key(second_key):
            sequence_a, sequence_b = first[2], second[2]
        else:
            sequence_a, sequence_b = second[2], first[2]
        index_a = self._kmer_index(sequence_a, reverse_complement_orientation=False)
        index_b = self._kmer_index(sequence_b, reverse_complement_orientation=True)
        result = cached_pair_chemistry(
            self.pair_cache,
            sequence_a,
            sequence_b,
            universe=self.universe,
            alpha=self.config.alpha,
            seed_length=self.config.binding_seed_length,
            minimum_length=self.config.minimum_binding_length,
            allow_gu_wobble=self.config.allow_gu_wobble,
            max_mismatches=self.config.max_mismatches,
            stacking_bonus=self.config.stacking_bonus,
            binding_energy_threshold=self.config.binding_energy_threshold,
            motif_length=self.config.reaction_motif_length,
            accessibility_a=self.accessibility_cache.get(sequence_a),
            accessibility_b=self.accessibility_cache.get(sequence_b),
            kmer_index_a=index_a,
            kmer_index_b=index_b,
            encounter_generation=self.generation,
            binding_lifetime_mode=self.config.binding_lifetime_mode,
            max_binding_lifetime=self.config.max_binding_lifetime,
            lifetime_energy_offset=self.config.lifetime_energy_offset,
            lifetime_temperature=self.config.lifetime_temperature,
        )
        return result

    def _kmer_index(
        self,
        sequence: ChemicalSequence,
        *,
        reverse_complement_orientation: bool,
    ) -> KmerIndex:
        """Return one cached forward/oriented k-mer index per species."""

        cache_key = (
            sequence.shape_key,
            self.config.binding_seed_length,
            reverse_complement_orientation,
        )
        index = self.kmer_index_cache.get(cache_key)
        if index is None:
            index = build_kmer_index(
                sequence.bases,
                self.config.binding_seed_length,
                reverse_complement_orientation=reverse_complement_orientation,
            )
            self.kmer_index_cache[cache_key] = index
        return index

    def _prepare_interactions(
        self,
        observations: list[tuple[Component, ShapeKey, ChemicalSequence]],
    ) -> tuple[list[PairChemistry], list[ActiveSite], np.ndarray, list[SiteEffect]]:
        pair_chemistries: dict[tuple[ShapeKey, ShapeKey], PairChemistry] = {}
        zone_items: list[tuple[SiteEffect, np.ndarray]] = []
        active_sites: list[ActiveSite] = []
        component_pairs = find_interacting_pairs(
            [item[0] for item in observations],
            (self.height, self.width),
            self.config.interaction_radius,
        )
        for first_index, second_index in component_pairs:
            first = observations[first_index]
            second = observations[second_index]
            chemistry = self._pair_chemistry(first, second)
            pair_chemistries[chemistry.pair_key] = chemistry
            if shape_key_sort_key(first[1]) <= shape_key_sort_key(second[1]):
                component_a, sequence_a = first[0], first[2]
                component_b, sequence_b = second[0], second[2]
            else:
                component_a, sequence_a = second[0], second[2]
                component_b, sequence_b = first[0], first[2]
            for site_interaction in chemistry.sites:
                effect = site_interaction_to_effect(
                    site_interaction,
                    self.base_rule,
                    max_rule_changes=self.config.site_max_rule_changes,
                    interaction_threshold=self.config.interaction_threshold,
                )
                zone = site_effect_zone(
                    component_a,
                    component_b,
                    sequence_a,
                    sequence_b,
                    site_interaction,
                    (self.height, self.width),
                    interaction_radius=self.config.interaction_radius,
                    effect_padding=self.config.effect_padding,
                    spatialize_sites=self.config.spatialize_sites,
                )
                active = ActiveSite(chemistry.pair_key, chemistry, effect, zone)
                active_sites.append(active)
                zone_items.append((effect, zone))
                chemistry.total_site_effect_area += int(np.count_nonzero(zone))
        owner_map, ordered_effects = resolve_site_owner_map(
            zone_items,
            grid_shape=(self.height, self.width),
        )
        if ordered_effects:
            self._effect_slot_capacity = max(self._effect_slot_capacity, len(ordered_effects))
        return list(pair_chemistries.values()), active_sites, owner_map, ordered_effects

    @staticmethod
    def _site_key(active: ActiveSite) -> tuple[Any, ...]:
        """Return a stable identity for one cached sequence binding site."""

        site = active.effect.interaction.site
        motif = active.effect.interaction.motif
        return (
            active.pair_key,
            site.start_a,
            site.end_a,
            site.start_b,
            site.end_b,
            motif.motif_length,
            motif.symmetric_codes,
        )

    def _resolve_active_sites(
        self,
        active_sites: list[ActiveSite],
    ) -> tuple[np.ndarray, list[SiteEffect]]:
        """Resolve a current/persisted site collection into JAX inputs."""

        owner_map, ordered_effects = resolve_site_owner_map(
            [(active.effect, active.zone) for active in active_sites],
            grid_shape=(self.height, self.width),
        )
        if ordered_effects:
            self._effect_slot_capacity = max(self._effect_slot_capacity, len(ordered_effects))
        return owner_map, ordered_effects

    def _apply_binding_lifetime(
        self,
        current_sites: list[ActiveSite],
    ) -> list[ActiveSite]:
        """Carry strong sites across deterministic transition lifetimes.

        ``remaining`` counts future transitions after the current one.  A
        refreshed site therefore applies immediately and can remain anchored
        to its last encounter zone for its deterministic energy-derived
        lifetime.  This mode is opt-in; the default instant mode has no
        persistent state.
        """

        if self.config.binding_lifetime_mode != "energy":
            self._bound_sites.clear()
            return current_sites
        previous = self._bound_sites
        current_by_key = {self._site_key(active): active for active in current_sites}
        combined = list(current_sites)
        next_bound: dict[tuple[Any, ...], tuple[ActiveSite, int]] = {}
        for key, (active, remaining) in previous.items():
            if key in current_by_key or remaining <= 0:
                continue
            combined.append(active)
            if remaining > 1:
                next_bound[key] = (active, remaining - 1)
        for key, active in current_by_key.items():
            remaining = max(0, int(active.effect.interaction.lifetime) - 1)
            if remaining:
                next_bound[key] = (active, remaining)
        self._bound_sites = next_bound
        return combined

    def step(self) -> StepResult:
        """Advance one generation in the fixed chemistry execution order."""

        previous = _host_grid(self.grid)
        source_generation = self.generation
        detection_performed = (
            source_generation >= self.config.warmup_steps
            and (source_generation - self.config.warmup_steps) % self.config.detect_every == 0
        )
        components: list[Component] = []
        observations: list[tuple[Component, ShapeKey, ChemicalSequence]] = []
        new_species_count = 0
        pair_chemistries: list[PairChemistry] = []
        active_sites: list[ActiveSite] = []
        ordered_effects: list[SiteEffect] = []
        owner_map = np.full((self.height, self.width), -1, dtype=np.int32)
        if detection_performed:
            components, observations, new_species_count = self._observe_current(previous)
            if self.config.interactions_enabled:
                pair_chemistries, active_sites, owner_map, ordered_effects = (
                    self._prepare_interactions(observations)
                )
        if self.config.binding_lifetime_mode == "energy":
            active_sites = self._apply_binding_lifetime(active_sites)
            owner_map, ordered_effects = self._resolve_active_sites(active_sites)
        self.metrics.observe_chemistry(
            observations=observations,
            pair_chemistries=pair_chemistries,
            active_sites=active_sites,
            detection_performed=detection_performed,
        )

        if ordered_effects:
            birth_masks, survival_masks = site_rule_arrays(
                ordered_effects,
                capacity=self._effect_slot_capacity,
            )
            next_grid = step_with_interactions(
                self.grid,
                owner_map,
                birth_masks,
                survival_masks,
                self.base_birth,
                self.base_survival,
            )
        else:
            next_grid = step_jit(self.grid, self.base_birth, self.base_survival)
        self.grid = next_grid
        self.generation += 1
        next_host = _host_grid(next_grid)
        metrics_row: dict[str, int | float] | None = None
        if self.generation % self.config.metrics_every == 0:
            metrics_row = self.metrics.record(
                generation=self.generation,
                previous_grid=previous,
                next_grid=next_host,
                components=components,
                observations=observations,
                new_species_count=new_species_count,
                pair_chemistries=pair_chemistries,
                active_sites=active_sites,
                owner_map=owner_map,
                detection_performed=detection_performed,
                chemistry_already_observed=True,
            )
        if self.generation % self.config.snapshot_every == 0:
            self.snapshots[self.generation] = next_host.copy()
        result = StepResult(
            source_generation=source_generation,
            generation=self.generation,
            grid=next_host.copy(),
            components=tuple(components),
            observations=tuple(observations),
            pair_chemistries=tuple(pair_chemistries),
            active_sites=tuple(active_sites),
            owner_map=owner_map.copy(),
            detection_performed=detection_performed,
            metrics=metrics_row,
        )
        self.last_step = result
        return result

    def run(self, *, progress: Callable[[int, int], None] | None = None) -> RNAExperimentResult:
        """Run the configured number of transitions."""

        for _ in range(self.config.steps):
            self.step()
            if progress is not None:
                progress(self.generation, self.config.steps)
        final = _host_grid(self.grid)
        self.snapshots.setdefault(self.generation, final.copy())
        return RNAExperimentResult(
            config=self.config,
            initial_grid=self.initial_grid.copy(),
            final_grid=final,
            universe=self.universe,
            species_registry=self.species_registry,
            pair_cache=self.pair_cache,
            motifs_by_species={
                key: len(values) for key, values in self.metrics.motifs_by_species.items()
            },
            generation_records=list(self.metrics.records),
            snapshots={generation: frame.copy() for generation, frame in self.snapshots.items()},
            summary=self.metrics.summary(
                final,
                species_count=len(self.species_registry),
                pair_count=len(self.pair_cache),
                last_generation=self.generation,
            ),
        )


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def default_output_directory(config: RNAExperimentConfig) -> Path:
    """Return the conventional timestamped artifact directory."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ARTIFACT_ROOT / f"{stamp}_seed{config.seed}_alpha{config.alpha:g}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(result: RNAExperimentResult, output_dir: str | Path | None = None) -> Path:
    """Write reproducible manifest, metrics, chemistry, and sparse snapshots."""

    target = default_output_directory(result.config) if output_dir is None else Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    initial = np.asarray(result.initial_grid, dtype=np.uint8)
    final = np.asarray(result.final_grid, dtype=np.uint8)
    _write_json(
        target / "manifest.json",
        {
            "experiment": "morphology_interactions/rna_chemistry",
            "version": 1,
            "config": result.config.as_dict(),
            "universe_seed": result.universe.seed,
            "base_rule": result.config.base_rule,
            "chemistry_calibration": result.universe.calibration_report(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "python_version": platform.python_version(),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "timestamp": datetime.now(UTC).isoformat(),
            "initial_grid_sha256": _sha256(initial),
            "final_grid_sha256": _sha256(final),
        },
    )
    summary = dict(result.summary)
    summary.update(
        {
            "grid_shape": [int(initial.shape[0]), int(initial.shape[1])],
            "initial_grid_sha256": _sha256(initial),
            "final_grid_sha256": _sha256(final),
        }
    )
    _write_json(target / "summary.json", summary)
    _write_csv(target / "generations.csv", GENERATION_FIELDS, result.generation_records)
    artifact_accessibility = AccessibilityCache(
        mode=result.config.accessibility_mode,
        minimum_hairpin_separation=result.config.minimum_hairpin_separation,
        paired_accessibility=result.config.paired_accessibility,
        allow_gu_wobble=result.config.allow_gu_wobble,
        max_nussinov_length=result.config.max_nussinov_length,
        window=result.config.fold_window,
    )
    species_rows = []
    for record in result.species_registry.records_by_id():
        frequencies = base_frequencies(record.sequence.bases)
        accessibility = artifact_accessibility.get(record.sequence)
        kmers = kmer_codes(record.sequence.bases, result.config.binding_seed_length)
        matrix = matrix_from_shape_key(record.key)
        species_rows.append(
            {
                "species_id": record.species_id,
                "height": record.key.height,
                "width": record.key.width,
                "cells": int(np.count_nonzero(matrix)),
                "perimeter": morphology_perimeter(record.key),
                "sequence_length": record.sequence.length,
                "sequence": record.sequence.text,
                "base_a_frequency": frequencies[0],
                "base_c_frequency": frequencies[1],
                "base_g_frequency": frequencies[2],
                "base_u_frequency": frequencies[3],
                "sequence_entropy": sequence_entropy(record.sequence.bases),
                "accessible_fraction": float(np.mean(accessibility)),
                "unique_kmer_count": int(np.unique(kmers).size),
                "unique_reaction_motif_count": result.motifs_by_species.get(record.key, 0),
                "first_seen": record.first_seen_generation,
                "last_seen": record.last_seen_generation,
                "observations": record.observations,
                "independent_components": record.independent_components,
            }
        )
    _write_csv(
        target / "species.csv",
        (
            "species_id",
            "height",
            "width",
            "cells",
            "perimeter",
            "sequence_length",
            "sequence",
            "base_a_frequency",
            "base_c_frequency",
            "base_g_frequency",
            "base_u_frequency",
            "sequence_entropy",
            "accessible_fraction",
            "unique_kmer_count",
            "unique_reaction_motif_count",
            "first_seen",
            "last_seen",
            "observations",
            "independent_components",
        ),
        species_rows,
    )
    species_ids = {key: record.species_id for key, record in result.species_registry.items()}
    interaction_rows = []
    for _, chemistry in sorted(result.pair_cache.items(), key=lambda item: item[0]):
        effects = [
            site_interaction_to_effect(
                site,
                result.config.base_rule,
                max_rule_changes=result.config.site_max_rule_changes,
                interaction_threshold=result.config.interaction_threshold,
            )
            for site in chemistry.sites
        ]
        distinct_rules = len({effect.rule_id for effect in effects})
        common = {
            "species_a": species_ids[chemistry.species_a],
            "species_b": species_ids[chemistry.species_b],
            "first_seen": chemistry.first_seen_generation,
            "candidate_seed_count": chemistry.candidate_seed_count,
            "raw_site_count": chemistry.raw_site_count,
            "successful_binding_sites": chemistry.successful_binding_sites,
            "encounters": chemistry.encounters,
            "total_paired_bases": chemistry.total_paired_bases,
            "total_site_effect_area": chemistry.total_site_effect_area,
            "distinct_site_rules": distinct_rules,
        }
        if not chemistry.sites:
            interaction_rows.append(common)
            continue
        for site_index, (site, effect) in enumerate(zip(chemistry.sites, effects, strict=True)):
            interaction_rows.append(
                {
                    **common,
                    "site_index": site_index,
                    "motif_code_a": site.motif.code_a,
                    "motif_code_b": site.motif.code_b,
                    "binding_score": site.binding_score,
                    "reaction_strength": site.strength,
                    "lifetime": site.lifetime,
                    "site_rule_id": effect.rule_id,
                    "local_rule": format_rule(effect.local_rule),
                }
            )
    _write_csv(
        target / "interactions.csv",
        (
            "species_a",
            "species_b",
            "first_seen",
            "candidate_seed_count",
            "raw_site_count",
            "successful_binding_sites",
            "encounters",
            "total_paired_bases",
            "total_site_effect_area",
            "distinct_site_rules",
            "site_index",
            "motif_code_a",
            "motif_code_b",
            "binding_score",
            "reaction_strength",
            "lifetime",
            "site_rule_id",
            "local_rule",
        ),
        interaction_rows,
    )
    _write_json(
        target / "species_keys.json",
        [
            {
                "species_id": record.species_id,
                "height": record.key.height,
                "width": record.key.width,
                "packed_hex": record.key.packed_hex,
                "sequence_length": record.sequence.length,
            }
            for record in result.species_registry.records_by_id()
        ],
    )
    generations = np.asarray(sorted(result.snapshots), dtype=np.int64)
    frames = np.asarray(
        [result.snapshots[int(generation)] for generation in generations], dtype=np.uint8
    )
    np.savez_compressed(target / "snapshots.npz", generations=generations, frames=frames)
    return target


def run_experiment(
    config: RNAExperimentConfig,
    *,
    initial_grid: Any | None = None,
    output_dir: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RNAExperimentResult:
    """Run one RNA chemistry universe and optionally persist its artifacts."""

    result = RNAChemistryEngine(config, initial_grid=initial_grid).run(progress=progress)
    write_artifacts(result, output_dir)
    return result


Engine = RNAChemistryEngine
