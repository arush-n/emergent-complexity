"""Deterministic, non-repeating trial keys and strategy scheduling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from ...encoding import deterministic_uint64
from ..config import RNAExperimentConfig


@dataclass(frozen=True)
class TrialStrategy:
    """One deterministic search strategy applied to a trial configuration."""

    key: str
    density: float
    alpha: float
    warmup_steps: int
    interaction_radius: int
    accessibility_mode: str
    binding_lifetime_mode: str

    def apply(self, base: RNAExperimentConfig) -> RNAExperimentConfig:
        """Return the resolved immutable config for this strategy."""

        return replace(
            base,
            density=self.density,
            alpha=self.alpha,
            warmup_steps=self.warmup_steps,
            interaction_radius=self.interaction_radius,
            accessibility_mode=self.accessibility_mode,
            binding_lifetime_mode=self.binding_lifetime_mode,
        )


STRATEGIES: tuple[TrialStrategy, ...] = (
    TrialStrategy("sparse_structured", 0.020, 0.0, 32, 2, "none", "energy"),
    TrialStrategy("roomy_structured", 0.040, 0.0, 64, 2, "none", "energy"),
    TrialStrategy("folded_structured", 0.030, 0.0, 32, 2, "simplified_fold", "energy"),
    TrialStrategy("mixed_contact", 0.045, 0.5, 32, 3, "none", "energy"),
    TrialStrategy("scrambled_instant", 0.025, 1.0, 64, 2, "none", "instant"),
)


@dataclass(frozen=True)
class TrialPlan:
    """All immutable inputs that identify and reproduce one environment."""

    trial: int
    schedule_index: int
    strategy: TrialStrategy
    config: RNAExperimentConfig
    config_key: str
    initial_seed: int
    trial_key: str

    def as_dict(self) -> dict[str, object]:
        """Return an event/manifest-safe representation."""

        return {
            "trial": self.trial,
            "schedule_index": self.schedule_index,
            "strategy_key": self.strategy.key,
            "config_key": self.config_key,
            "initial_seed": self.initial_seed,
            "trial_key": self.trial_key,
            "config": self.config.as_dict(),
        }


def config_key(config: RNAExperimentConfig) -> str:
    """Hash the resolved config with stable JSON, never Python ``hash``."""

    payload = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(payload, digest_size=12, person=b"rna-config-v1").hexdigest()


def initial_seed(universe_seed: int, trial: int) -> int:
    """Derive a unique-keyed deterministic soup seed for a trial."""

    return deterministic_uint64(universe_seed, "rna-search-initial-state", trial) & 0x7FFFFFFF


def plan_for_trial(
    base: RNAExperimentConfig,
    *,
    trial: int,
    schedule_index: int,
    mode: str = "rotating",
    previous_strategy: str | None = None,
) -> TrialPlan:
    """Create a trial plan; rotating mode avoids the previous strategy."""

    if mode not in {"fixed", "rotating"}:
        raise ValueError("mode must be fixed or rotating")
    if trial < 0 or schedule_index < 0:
        raise ValueError("trial and schedule_index must be non-negative")
    if mode == "fixed":
        strategy = TrialStrategy(
            "fixed",
            base.density,
            base.alpha,
            base.warmup_steps,
            base.interaction_radius,
            base.accessibility_mode,
            base.binding_lifetime_mode,
        )
    else:
        strategy = STRATEGIES[schedule_index % len(STRATEGIES)]
        if previous_strategy is not None and strategy.key == previous_strategy:
            strategy = STRATEGIES[(schedule_index + 1) % len(STRATEGIES)]
    resolved = strategy.apply(base)
    resolved_key = config_key(resolved)
    seed = initial_seed(base.seed, trial)
    trial_key = f"{resolved_key}:{seed:08x}:{trial:016x}"
    return TrialPlan(
        trial=trial,
        schedule_index=schedule_index,
        strategy=strategy,
        config=resolved,
        config_key=resolved_key,
        initial_seed=seed,
        trial_key=trial_key,
    )


def schedule_manifest(base: RNAExperimentConfig, mode: str) -> dict[str, object]:
    """Describe the scheduler without embedding runtime state."""

    return {
        "mode": mode,
        "profiles": [
            {
                "key": strategy.key,
                "density": strategy.density,
                "alpha": strategy.alpha,
                "warmup_steps": strategy.warmup_steps,
                "interaction_radius": strategy.interaction_radius,
                "accessibility_mode": strategy.accessibility_mode,
                "binding_lifetime_mode": strategy.binding_lifetime_mode,
            }
            for strategy in STRATEGIES
        ],
        "fixed_base_config_key": config_key(base),
        "initial_seed_namespace": "rna-search-initial-state",
    }
