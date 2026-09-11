"""Map bounded site reactions onto spatially local Life-rule effects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ....core.rules import Rule, format_rule, rule_to_masks
from ..canonical import (
    matrix_from_shape_key,
    unwrap_toroidal_coordinates,
)
from ..components import Component
from ..interaction import build_interaction_zone, interaction_vector_to_rule
from .chemistry import SiteInteraction
from .sequence import ChemicalSequence


@dataclass(frozen=True)
class SiteEffect:
    """One bounded local B/S rule derived from one chemical binding site."""

    interaction: SiteInteraction
    local_rule: Rule
    rule_id: int
    birth_mask: np.ndarray
    survival_mask: np.ndarray

    def __post_init__(self) -> None:
        birth = np.asarray(self.birth_mask, dtype=np.uint8)
        survival = np.asarray(self.survival_mask, dtype=np.uint8)
        if birth.shape != (9,) or survival.shape != (9,):
            raise ValueError("site rule masks must have shape (9,)")
        birth.setflags(write=False)
        survival.setflags(write=False)
        object.__setattr__(self, "birth_mask", birth)
        object.__setattr__(self, "survival_mask", survival)

    @property
    def strength(self) -> float:
        """Return the continuous reaction strength for overlap resolution."""

        return self.interaction.strength

    @property
    def local_rule_text(self) -> str:
        """Return the canonical local rule text."""

        return format_rule(self.local_rule)

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Stable identity used for equal-strength site overlaps."""

        motif = self.interaction.motif
        site = self.interaction.site
        return (
            motif.symmetric_codes,
            site.start_a,
            site.start_b,
            site.end_a,
            site.end_b,
            self.rule_id,
        )


def site_interaction_to_effect(
    interaction: SiteInteraction,
    base_rule: Rule | str = "B3/S23",
    *,
    max_rule_changes: int = 1,
    interaction_threshold: float = 0.35,
) -> SiteEffect:
    """Convert one site vector to a small deterministic local Life rule."""

    local_rule, rule_id = interaction_vector_to_rule(
        interaction.final_vector,
        base_rule,
        max_rule_changes=max_rule_changes,
        interaction_threshold=interaction_threshold,
    )
    birth, survival = rule_to_masks(local_rule)
    return SiteEffect(
        interaction=interaction,
        local_rule=local_rule,
        rule_id=rule_id,
        birth_mask=np.asarray(birth, dtype=np.uint8),
        survival_mask=np.asarray(survival, dtype=np.uint8),
    )


def _transformed_coordinate(
    shape: tuple[int, int],
    coordinate: tuple[int, int],
    rotation: int,
    reflected: bool,
) -> tuple[int, int]:
    """Apply the same transform used by ``np.rot90``/``np.fliplr``."""

    row, col = coordinate
    height, width = shape
    for _ in range(rotation % 4):
        row, col = width - 1 - col, row
        height, width = width, height
    if reflected:
        col = width - 1 - col
    return row, col


def component_site_coordinates(
    component: Component,
    sequence: ChemicalSequence,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Map every sequence position to a toroidal world coordinate once.

    The returned ``(length, 2)`` array uses ``(-1, -1)`` for abstract or
    otherwise unmappable positions.  Computing the canonical transform once
    per component is important when one encounter exposes many binding sites;
    the previous scalar helper repeated the same matrix comparisons for every
    nucleotide in every site.
    """

    values = unwrap_toroidal_coordinates(component.coordinates, grid_shape)
    root = tuple(
        int(value)
        for value in component.coordinates[
            np.argmin(component.coordinates[:, 0] * grid_shape[1] + component.coordinates[:, 1])
        ]
    )
    local_min = values.min(axis=0)
    local_values = values - local_min
    local_shape = (int(local_values[:, 0].max()) + 1, int(local_values[:, 1].max()) + 1)
    canonical = matrix_from_shape_key(sequence.shape_key)
    target = np.zeros(local_shape, dtype=bool)
    target[local_values[:, 0], local_values[:, 1]] = True
    valid_positions = np.all(sequence.site_coordinates >= 0, axis=1)
    mapped = np.full((sequence.length, 2), -1, dtype=np.int64)
    if not np.any(valid_positions):
        mapped.setflags(write=False)
        return mapped

    for rotation in range(4):
        rotated = np.rot90(canonical, k=rotation).copy()
        candidates = ((rotated, False), (np.fliplr(rotated).copy(), True))
        for candidate, reflected in candidates:
            if candidate.shape != local_shape or not np.array_equal(candidate != 0, target):
                continue
            for position in np.flatnonzero(valid_positions):
                transformed_site = _transformed_coordinate(
                    canonical.shape,
                    tuple(int(value) for value in sequence.site_coordinates[position]),
                    rotation,
                    reflected,
                )
                unwrapped = np.asarray(transformed_site, dtype=np.int64) + local_min
                mapped[position] = (
                    (root[0] + int(unwrapped[0])) % int(grid_shape[0]),
                    (root[1] + int(unwrapped[1])) % int(grid_shape[1]),
                )
            mapped.setflags(write=False)
            return mapped
    mapped.setflags(write=False)
    return mapped


def component_site_coordinate(
    component: Component,
    sequence: ChemicalSequence,
    position: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Map one sequence position to a toroidal world coordinate when possible."""

    if position < 0 or position >= sequence.length:
        return None
    coordinate = component_site_coordinates(component, sequence, grid_shape)[position]
    if coordinate[0] < 0 or coordinate[1] < 0:
        return None
    return int(coordinate[0]), int(coordinate[1])


@lru_cache(maxsize=16)
def _dilation_offsets(radius: int) -> np.ndarray:
    """Cache the small Chebyshev offset stencil used by site zones."""

    values = np.asarray(
        [(row, col) for row in range(-radius, radius + 1) for col in range(-radius, radius + 1)],
        dtype=np.int64,
    )
    values.setflags(write=False)
    return values


def site_effect_zone(
    component_a: Component,
    component_b: Component,
    sequence_a: ChemicalSequence,
    sequence_b: ChemicalSequence,
    interaction: SiteInteraction,
    grid_shape: tuple[int, int],
    *,
    interaction_radius: int = 2,
    effect_padding: int = 1,
    spatialize_sites: bool = True,
    encounter_zone: np.ndarray | None = None,
    mapped_coordinates_a: np.ndarray | None = None,
    mapped_coordinates_b: np.ndarray | None = None,
) -> np.ndarray:
    """Anchor one site effect near its mapped sequence positions.

    If a canonical-to-world transform cannot be recovered (for example for a
    deliberately abstract exact-shape header position), the function falls
    back to the complete encounter zone.  This preserves Stage 1 chemistry
    validation while supporting Stage 2 spatialized sites whenever possible.
    """

    if encounter_zone is None:
        encounter_zone = build_interaction_zone(
            component_a,
            component_b,
            grid_shape,
            interaction_radius=interaction_radius,
            effect_padding=effect_padding,
        )
    else:
        encounter_zone = np.asarray(encounter_zone, dtype=bool)
        if encounter_zone.shape != grid_shape:
            raise ValueError("encounter_zone must match grid_shape")
    if not spatialize_sites:
        return encounter_zone
    site = interaction.site
    first_coordinates = (
        component_site_coordinates(component_a, sequence_a, grid_shape)
        if mapped_coordinates_a is None
        else np.asarray(mapped_coordinates_a, dtype=np.int64)
    )
    second_coordinates = (
        component_site_coordinates(component_b, sequence_b, grid_shape)
        if mapped_coordinates_b is None
        else np.asarray(mapped_coordinates_b, dtype=np.int64)
    )
    anchor_rows = np.concatenate(
        (
            first_coordinates[site.start_a : site.end_a, 0],
            second_coordinates[site.start_b : site.end_b, 0],
        )
    )
    anchor_cols = np.concatenate(
        (
            first_coordinates[site.start_a : site.end_a, 1],
            second_coordinates[site.start_b : site.end_b, 1],
        )
    )
    valid = (anchor_rows >= 0) & (anchor_cols >= 0)
    if not np.any(valid):
        return encounter_zone
    # The old implementation materialized a full-grid anchor mask and then
    # performed (2r+1)^2 full-grid rolls for every site.  Binding sites are
    # short and sparse, so enumerate their bounded toroidal neighborhoods and
    # write only those coordinates.  This is exactly the same Chebyshev
    # dilation, followed by the same encounter-zone intersection, without a
    # full-grid scan per site.
    radius = interaction_radius + effect_padding
    offsets = _dilation_offsets(radius)
    rows = (anchor_rows[valid, None] + offsets[None, :, 0]).reshape(-1) % int(grid_shape[0])
    cols = (anchor_cols[valid, None] + offsets[None, :, 1]).reshape(-1) % int(grid_shape[1])
    localized = np.zeros(grid_shape, dtype=bool)
    localized[rows, cols] = True
    localized &= encounter_zone
    return localized if np.any(localized) else encounter_zone


def resolve_site_owner_map(
    zones: list[tuple[SiteEffect, Any]],
    *,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, list[SiteEffect]]:
    """Resolve overlapping site zones by strength, then chemical identity."""

    height, width = grid_shape
    merged: dict[tuple[Any, ...], tuple[SiteEffect, np.ndarray]] = {}
    for effect, zone in zones:
        values = np.asarray(zone, dtype=bool)
        if values.shape != (height, width):
            raise ValueError("site zones must match grid_shape")
        key = effect.sort_key
        if key in merged:
            merged[key] = (effect, merged[key][1] | values)
        else:
            merged[key] = (effect, values.copy())
    ordered = [
        item[0]
        for item in sorted(
            merged.values(),
            key=lambda item: item[0].sort_key,
        )
        if np.any(item[1])
    ]
    zone_by_key = {effect.sort_key: merged[effect.sort_key][1] for effect in ordered}
    owner_map = np.full((height, width), -1, dtype=np.int32)
    best_strength = np.full((height, width), -np.inf, dtype=np.float32)
    for index, effect in enumerate(ordered):
        zone = zone_by_key[effect.sort_key]
        replace = zone & (float(effect.strength) > best_strength)
        owner_map[replace] = index
        best_strength[replace] = float(effect.strength)
    return owner_map, ordered


def site_rule_arrays(
    effects: list[SiteEffect],
    *,
    capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack site rules for V1's one-pass JAX local step."""

    if capacity is None:
        table_size = len(effects)
    else:
        if not isinstance(capacity, int) or capacity < len(effects) or capacity < 1:
            raise ValueError("capacity must contain every site effect and be positive")
        table_size = capacity
    birth = np.zeros((table_size, 9), dtype=np.uint8)
    survival = np.zeros((table_size, 9), dtype=np.uint8)
    if effects:
        birth[: len(effects)] = np.asarray(
            [effect.birth_mask for effect in effects],
            dtype=np.uint8,
        )
        survival[: len(effects)] = np.asarray(
            [effect.survival_mask for effect in effects],
            dtype=np.uint8,
        )
    return birth, survival


def effect_rule_ids(effects: list[SiteEffect]) -> set[int]:
    """Return the distinct local rules represented by a site collection."""

    return {int(effect.rule_id) for effect in effects}
