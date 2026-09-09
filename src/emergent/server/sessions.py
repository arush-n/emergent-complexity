"""Lightweight in-memory session management for the local simulator UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from ..core.grid import Grid, validate_dimensions
from ..core.measurements import alive_count, transition_counts
from ..core.random import key_from_seed, random_grid
from ..core.random import random_rule as make_random_rule
from ..core.rules import Rule, format_rule, parse_rule
from ..core.simulate import (
    run_steps,
    run_steps_dynamic_with_metrics,
    run_steps_with_metrics,
)
from ..core.simulate import run_until_stable_with_metrics as run_until_stable_jax
from .session_store import SessionLifecycle, resolve_seed


@dataclass
class SimulationSession:
    """Mutable UI/session state around an otherwise pure JAX grid."""

    grid: Grid
    initial_grid: Grid
    rule: Rule
    generation: int = 0
    seed: int = 42
    initial_density: float = 0.20
    changed_cells: int = 0
    births: int = 0
    deaths: int = 0

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])


class SessionStore(SessionLifecycle):
    """A bounded, expiring in-memory store for one server process."""

    def __init__(
        self,
        *,
        max_sessions: int = 256,
        session_ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._sessions: dict[str, SimulationSession] = {}
        self._init_lifecycle(
            max_sessions=max_sessions,
            session_ttl_seconds=session_ttl_seconds,
            clock=clock,
        )

    def create(
        self,
        *,
        width: int,
        height: int,
        density: float,
        seed: int | None,
        rule: Rule | str,
        session_id: str | None = None,
    ) -> str:
        """Create or replace a session with a deterministic random initial state."""

        validate_dimensions(height, width)
        parsed_rule = parse_rule(rule) if isinstance(rule, str) else rule
        actual_seed = resolve_seed(seed)
        grid = random_grid(key_from_seed(actual_seed), height, width, density)
        identifier = session_id or uuid4().hex
        self._sessions[identifier] = SimulationSession(
            grid=grid,
            initial_grid=grid,
            rule=parsed_rule,
            seed=actual_seed,
            initial_density=float(density),
        )
        self._touch(identifier)
        self._prune(protected_id=identifier)
        return identifier

    def ensure_default(self) -> str:
        """Create the default session on first API access."""

        self._prune()
        if "default" not in self._sessions:
            self.create(
                width=128,
                height=128,
                density=0.20,
                seed=42,
                rule="B3/S23",
                session_id="default",
            )
        self._touch("default")
        return "default"

    def get(self, session_id: str = "default") -> SimulationSession:
        """Return a session or raise ``KeyError`` if it does not exist."""

        self._prune()
        if session_id == "default" and session_id not in self._sessions:
            self.ensure_default()
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session: {session_id}") from exc
        self._touch(session_id)
        return session

    def reset(self, session_id: str = "default") -> SimulationSession:
        """Restore the most recent initial grid and generation zero."""

        session = self.get(session_id)
        session.grid = session.initial_grid
        session.generation = 0
        session.changed_cells = 0
        session.births = 0
        session.deaths = 0
        return session

    def clear(self, session_id: str = "default") -> SimulationSession:
        """Clear the current state while retaining the reset target."""

        session = self.get(session_id)
        session.grid = jnp.zeros_like(session.grid)
        session.generation = 0
        session.changed_cells = 0
        session.births = 0
        session.deaths = 0
        return session

    def randomize(
        self,
        session_id: str = "default",
        *,
        seed: int | None = 42,
        density: float = 0.20,
    ) -> SimulationSession:
        """Create and retain a new deterministic initial state."""

        session = self.get(session_id)
        actual_seed = resolve_seed(seed)
        grid = random_grid(
            key_from_seed(actual_seed),
            session.height,
            session.width,
            density,
        )
        session.grid = grid
        session.initial_grid = grid
        session.seed = actual_seed
        session.initial_density = float(density)
        session.generation = 0
        session.changed_cells = 0
        session.births = 0
        session.deaths = 0
        return session

    def set_rule(self, rule: Rule | str, session_id: str = "default") -> SimulationSession:
        """Change the transition rule without changing the current grid."""

        session = self.get(session_id)
        session.rule = parse_rule(rule) if isinstance(rule, str) else rule
        return session

    def random_rule(
        self,
        session_id: str = "default",
        *,
        seed: int | None = None,
    ) -> SimulationSession:
        """Set a reproducible random rule using an explicit seed."""

        session = self.get(session_id)
        actual_seed = session.seed if seed is None else resolve_seed(seed)
        session.rule = make_random_rule(key_from_seed(actual_seed))
        return session

    def step(
        self,
        session_id: str = "default",
        *,
        steps: int = 1,
        collect_metrics: bool = False,
    ) -> list[dict[str, int | float]]:
        """Advance the session and optionally return per-generation metrics.

        The frontend uses ``collect_metrics`` when it advances several
        generations between renders. The numerical work stays in a JAX scan;
        the returned list is only the small JSON representation needed by the
        chart.
        """

        session = self.get(session_id)
        start_generation = session.generation
        if collect_metrics:
            final_grid, metric_values = run_steps_with_metrics(session.grid, session.rule, steps)
            host_metrics = jax.device_get(metric_values)
            session.grid = final_grid
            metrics = [
                self._metric_payload(
                    row,
                    generation=start_generation + index + 1,
                    total_cells=session.height * session.width,
                )
                for index, row in enumerate(host_metrics)
            ]
            if metrics:
                last = metrics[-1]
                session.changed_cells = int(last["changed_cells"])
                session.births = int(last["births"])
                session.deaths = int(last["deaths"])
        elif steps != 1:
            final_grid, transition = run_steps_dynamic_with_metrics(
                session.grid,
                session.rule,
                steps,
            )
            session.grid = final_grid
            host_transition = jax.device_get(transition)
            session.changed_cells = int(host_transition[1])
            session.births = int(host_transition[2])
            session.deaths = int(host_transition[3])
            metrics = []
        else:
            previous_grid = session.grid
            session.grid = run_steps(previous_grid, session.rule, steps)
            transition = jax.device_get(transition_counts(previous_grid, session.grid))
            session.changed_cells = int(transition[1])
            session.births = int(transition[2])
            session.deaths = int(transition[3])
            metrics = []
        session.generation += int(steps)
        return metrics

    def run_until_stable(
        self,
        session_id: str = "default",
        *,
        max_steps: int = 1_000,
    ) -> tuple[int, bool]:
        """Advance until a fixed point or return after the safety bound."""

        session = self.get(session_id)
        final_grid, steps_taken, settled, last_metrics = run_until_stable_jax(
            session.grid,
            session.rule,
            max_steps,
        )
        session.grid = final_grid
        session.generation += int(steps_taken)
        host_metrics = jax.device_get(last_metrics)
        session.changed_cells = int(host_metrics[1])
        session.births = int(host_metrics[2])
        session.deaths = int(host_metrics[3])
        return int(steps_taken), bool(settled)

    def set_state(
        self,
        grid: Any,
        session_id: str = "default",
        *,
        rule: Rule | str | None = None,
        seed: int | None = None,
        density: float | None = None,
        generation: int = 0,
    ) -> SimulationSession:
        """Replace the current state and make it the new reset target."""

        session = self.get(session_id)
        values = jnp.asarray(grid, dtype=jnp.uint8)
        if values.ndim != 2 or values.shape != session.grid.shape:
            raise ValueError(f"grid shape must be {session.grid.shape}, got {values.shape}")
        values = (values != 0).astype(jnp.uint8)
        session.grid = values
        session.initial_grid = values
        if rule is not None:
            session.rule = parse_rule(rule) if isinstance(rule, str) else rule
        if seed is not None:
            session.seed = seed
        if density is not None:
            session.initial_density = float(density)
        session.generation = int(generation)
        # Drawing edits are not generations, so keep activity graphs focused on
        # transitions produced by the cellular-automaton rule.
        session.changed_cells = 0
        session.births = 0
        session.deaths = 0
        return session

    @staticmethod
    def _metric_payload(
        values: Any,
        *,
        generation: int,
        total_cells: int,
    ) -> dict[str, int | float]:
        """Convert one JAX transition-metric row into JSON-compatible values."""

        alive, changed, births, deaths = (int(value) for value in values)
        return {
            "generation": generation,
            "alive": alive,
            "alive_fraction": alive / total_cells,
            "changed_cells": changed,
            "changed_fraction": changed / total_cells,
            "births": births,
            "birth_fraction": births / total_cells,
            "deaths": deaths,
            "death_fraction": deaths / total_cells,
        }

    def payload(self, session_id: str = "default", *, include_grid: bool = False) -> dict[str, Any]:
        """Return state metadata; include the full grid only for explicit exports."""

        session = self.get(session_id)
        total_cells = session.height * session.width
        alive = int(alive_count(session.grid))
        response: dict[str, Any] = {
            "session_id": session_id,
            "rule": format_rule(session.rule),
            "width": session.width,
            "height": session.height,
            "seed": session.seed,
            "initial_density": session.initial_density,
            "generation": session.generation,
            "alive": alive,
            "alive_fraction": alive / total_cells,
            "changed_cells": session.changed_cells,
            "changed_fraction": session.changed_cells / total_cells,
            "births": session.births,
            "birth_fraction": session.births / total_cells,
            "deaths": session.deaths,
            "death_fraction": session.deaths / total_cells,
        }
        if include_grid:
            response["grid"] = np.asarray(jax.device_get(session.grid), dtype=np.uint8).tolist()
        return response

    def render_bytes(self, session_id: str = "default") -> tuple[bytes, int, int, int]:
        """Return the current grid packed one bit per cell for browser rendering."""

        session = self.get(session_id)
        values = np.asarray(jax.device_get(session.grid), dtype=np.uint8)
        packed = np.packbits(values.reshape(-1), bitorder="little")
        return packed.tobytes(), session.width, session.height, session.generation
