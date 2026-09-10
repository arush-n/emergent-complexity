"""Configuration for the isolated RNA-inspired chemistry experiment."""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from typing import Any

from ....core.rules import Rule, format_rule, parse_rule


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probability(value: float, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class RNAExperimentConfig:
    """Resolved immutable parameters for one RNA-chemistry universe."""

    width: int = 128
    height: int = 128
    seed: int = 42
    density: float = 0.10
    base_rule: str = "B3/S23"
    steps: int = 5_000
    warmup_steps: int = 100
    min_component_cells: int = 1
    rotation_invariant: bool = True
    reflection_invariant: bool = False

    sequence_mode: str = "local_surface"
    binding_seed_length: int = 4
    minimum_binding_length: int = 4
    allow_gu_wobble: bool = True
    max_mismatches: int = 1
    stacking_bonus: float = -0.5
    binding_energy_threshold: float = -5.0

    reaction_motif_length: int = 4
    beta: float = 1.0
    calibration_size: int = 512
    site_max_rule_changes: int = 1
    interaction_threshold: float = 0.35
    interaction_radius: int = 2
    effect_padding: int = 1

    accessibility_mode: str = "none"
    minimum_hairpin_separation: int = 3
    paired_accessibility: float = 0.25
    max_nussinov_length: int = 256
    fold_window: int = 64
    binding_lifetime_mode: str = "instant"
    max_binding_lifetime: int = 16
    lifetime_energy_offset: float = 5.0
    lifetime_temperature: float = 2.0

    alpha: float = 0.0
    spatialize_sites: bool = True
    metrics_every: int = 1
    snapshot_every: int = 100
    detect_every: int = 1
    component_backend: str = "auto"

    interactions_enabled: bool = True
    initial_condition: str = "random"
    pattern_name: str = "glider"
    pattern_row: int | None = None
    pattern_col: int | None = None
    initial_condition_path: str | None = None
    native_base_url: str = "http://127.0.0.1:8000"

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _integer(self.width, "width", minimum=1))
        object.__setattr__(self, "height", _integer(self.height, "height", minimum=1))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(self, "density", _probability(self.density, "density"))
        object.__setattr__(self, "steps", _integer(self.steps, "steps"))
        object.__setattr__(self, "warmup_steps", _integer(self.warmup_steps, "warmup_steps"))
        object.__setattr__(
            self,
            "min_component_cells",
            _integer(self.min_component_cells, "min_component_cells", minimum=1),
        )
        if not isinstance(self.base_rule, str):
            raise TypeError("base_rule must be a string")
        object.__setattr__(self, "base_rule", format_rule(parse_rule(self.base_rule)))

        if self.sequence_mode not in {"local_surface", "exact_shape"}:
            raise ValueError("sequence_mode must be local_surface or exact_shape")
        for name in ("rotation_invariant", "reflection_invariant"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.reflection_invariant and not self.rotation_invariant:
            raise ValueError("reflection_invariant requires rotation_invariant")
        for name in (
            "binding_seed_length",
            "minimum_binding_length",
            "reaction_motif_length",
            "minimum_hairpin_separation",
            "max_binding_lifetime",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        object.__setattr__(self, "max_mismatches", _integer(self.max_mismatches, "max_mismatches"))
        if self.minimum_binding_length < self.binding_seed_length:
            raise ValueError("minimum_binding_length must be at least binding_seed_length")
        if not isinstance(self.stacking_bonus, (int, float)):
            raise TypeError("stacking_bonus must be a number")
        object.__setattr__(self, "stacking_bonus", float(self.stacking_bonus))
        if not isinstance(self.binding_energy_threshold, (int, float)):
            raise TypeError("binding_energy_threshold must be a number")
        object.__setattr__(self, "binding_energy_threshold", float(self.binding_energy_threshold))
        if not isinstance(self.beta, (int, float)) or float(self.beta) < 0.0:
            raise ValueError("beta must be a non-negative number")
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(
            self,
            "calibration_size",
            _integer(self.calibration_size, "calibration_size", minimum=32),
        )
        object.__setattr__(
            self,
            "site_max_rule_changes",
            _integer(self.site_max_rule_changes, "site_max_rule_changes"),
        )
        object.__setattr__(
            self,
            "interaction_threshold",
            _probability(self.interaction_threshold, "interaction_threshold"),
        )
        object.__setattr__(
            self, "interaction_radius", _integer(self.interaction_radius, "interaction_radius")
        )
        object.__setattr__(self, "effect_padding", _integer(self.effect_padding, "effect_padding"))
        if self.accessibility_mode not in {"none", "simplified_fold"}:
            raise ValueError("accessibility_mode must be none or simplified_fold")
        object.__setattr__(
            self,
            "paired_accessibility",
            _probability(self.paired_accessibility, "paired_accessibility"),
        )
        object.__setattr__(
            self,
            "max_nussinov_length",
            _integer(self.max_nussinov_length, "max_nussinov_length", minimum=1),
        )
        object.__setattr__(
            self, "fold_window", _integer(self.fold_window, "fold_window", minimum=1)
        )
        if self.binding_lifetime_mode not in {"instant", "energy"}:
            raise ValueError("binding_lifetime_mode must be instant or energy")
        if not isinstance(self.lifetime_energy_offset, (int, float)):
            raise TypeError("lifetime_energy_offset must be a number")
        if (
            not isinstance(self.lifetime_temperature, (int, float))
            or float(self.lifetime_temperature) <= 0
        ):
            raise ValueError("lifetime_temperature must be positive")
        object.__setattr__(self, "lifetime_energy_offset", float(self.lifetime_energy_offset))
        object.__setattr__(self, "lifetime_temperature", float(self.lifetime_temperature))
        object.__setattr__(self, "alpha", _probability(self.alpha, "alpha"))
        for name in ("allow_gu_wobble", "spatialize_sites", "interactions_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(
            self, "metrics_every", _integer(self.metrics_every, "metrics_every", minimum=1)
        )
        object.__setattr__(
            self, "snapshot_every", _integer(self.snapshot_every, "snapshot_every", minimum=1)
        )
        object.__setattr__(
            self, "detect_every", _integer(self.detect_every, "detect_every", minimum=1)
        )
        if self.component_backend not in {"auto", "python", "scipy"}:
            raise ValueError("component_backend must be auto, python, or scipy")
        if self.initial_condition not in {"random", "patterns", "npz", "api"}:
            raise ValueError("initial_condition must be random, patterns, npz, or api")
        if self.initial_condition == "npz" and not self.initial_condition_path:
            raise ValueError("initial_condition_path is required for initial_condition='npz'")
        if self.initial_condition == "api" and not self.native_base_url:
            raise ValueError("native_base_url is required for initial_condition='api'")
        if self.pattern_row is not None:
            object.__setattr__(self, "pattern_row", _integer(self.pattern_row, "pattern_row"))
        if self.pattern_col is not None:
            object.__setattr__(self, "pattern_col", _integer(self.pattern_col, "pattern_col"))

    @property
    def rule(self) -> Rule:
        """Return the parsed native rule."""

        return parse_rule(self.base_rule)

    def as_dict(self) -> dict[str, Any]:
        """Return the resolved configuration in JSON-compatible form."""

        return asdict(self)
