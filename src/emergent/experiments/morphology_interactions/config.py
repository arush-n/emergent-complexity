"""Configuration for the deterministic morphology-interaction experiment."""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from typing import Any

from ...core.rules import Rule, format_rule, parse_rule


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    """Return an integer-like value after applying an experiment bound."""

    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probability(value: float, name: str) -> float:
    """Validate a probability-like parameter without changing its value."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class MorphologyExperimentConfig:
    """Resolved, immutable parameters for one experiment universe.

    ``seed`` serves both as the native deterministic initial-condition seed and
    as the universe seed for the fixed encoding/interaction law.  A caller can
    pass an explicit ``initial_grid`` to the engine when those two concerns
    need to be separated while retaining the same universe law.
    """

    width: int = 128
    height: int = 128

    seed: int = 42
    density: float = 0.10

    base_rule: str = "B3/S23"

    steps: int = 5_000
    warmup_steps: int = 100

    min_component_cells: int = 1
    component_backend: str = "auto"

    rotation_invariant: bool = True
    reflection_invariant: bool = False

    identity_dim: int = 32

    alpha: float = 0.5
    beta: float = 1.0

    interaction_radius: int = 2
    effect_padding: int = 1

    max_rule_changes: int = 2
    interaction_threshold: float = 0.35

    metrics_every: int = 1
    snapshot_every: int = 100
    detect_every: int = 1

    interactions_enabled: bool = True
    initial_condition: str = "random"
    pattern_name: str = "glider"
    pattern_row: int | None = None
    pattern_col: int | None = None
    initial_condition_path: str | None = None
    native_base_url: str = "http://127.0.0.1:8000"

    # Version 1 deliberately has one physical effect.  The field is explicit
    # so a future operator effect cannot be mixed silently with these results.
    effect_mode: str = "local_rule"

    def __post_init__(self) -> None:
        """Validate all values at construction time."""

        object.__setattr__(self, "width", _integer(self.width, "width", minimum=1))
        object.__setattr__(self, "height", _integer(self.height, "height", minimum=1))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(self, "density", _probability(self.density, "density"))
        object.__setattr__(self, "steps", _integer(self.steps, "steps"))
        object.__setattr__(
            self,
            "warmup_steps",
            _integer(self.warmup_steps, "warmup_steps"),
        )
        object.__setattr__(
            self,
            "min_component_cells",
            _integer(self.min_component_cells, "min_component_cells", minimum=1),
        )
        if not isinstance(self.component_backend, str) or self.component_backend not in {
            "auto",
            "python",
            "scipy",
        }:
            raise ValueError("component_backend must be one of: auto, python, scipy")
        object.__setattr__(
            self,
            "identity_dim",
            _integer(self.identity_dim, "identity_dim", minimum=6),
        )
        object.__setattr__(self, "alpha", _probability(self.alpha, "alpha"))
        if not isinstance(self.beta, (int, float)):
            raise TypeError("beta must be a number")
        if float(self.beta) < 0.0:
            raise ValueError("beta must be non-negative")
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(
            self,
            "interaction_radius",
            _integer(self.interaction_radius, "interaction_radius"),
        )
        object.__setattr__(
            self,
            "effect_padding",
            _integer(self.effect_padding, "effect_padding"),
        )
        object.__setattr__(
            self,
            "max_rule_changes",
            _integer(self.max_rule_changes, "max_rule_changes"),
        )
        object.__setattr__(
            self,
            "interaction_threshold",
            _probability(self.interaction_threshold, "interaction_threshold"),
        )
        object.__setattr__(
            self,
            "metrics_every",
            _integer(self.metrics_every, "metrics_every", minimum=1),
        )
        object.__setattr__(
            self,
            "snapshot_every",
            _integer(self.snapshot_every, "snapshot_every", minimum=1),
        )
        object.__setattr__(
            self,
            "detect_every",
            _integer(self.detect_every, "detect_every", minimum=1),
        )

        if not isinstance(self.base_rule, str):
            raise TypeError("base_rule must be a string")
        object.__setattr__(self, "base_rule", format_rule(parse_rule(self.base_rule)))

        for name in ("rotation_invariant", "reflection_invariant", "interactions_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.reflection_invariant and not self.rotation_invariant:
            # Reflection-only is mathematically well-defined, but not one of
            # the three documented comparison modes, so reject ambiguity.
            raise ValueError("reflection_invariant requires rotation_invariant")

        valid_conditions = {"random", "patterns", "npz", "api"}
        if self.initial_condition not in valid_conditions:
            choices = ", ".join(sorted(valid_conditions))
            raise ValueError(f"initial_condition must be one of: {choices}")
        if self.effect_mode != "local_rule":
            raise ValueError("version 1 supports only effect_mode='local_rule'")
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
        """Return the parsed native rule used by this universe."""

        return parse_rule(self.base_rule)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration mapping."""

        return asdict(self)
