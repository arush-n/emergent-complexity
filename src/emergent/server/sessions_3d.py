"""Lightweight in-memory sessions for the 3D browser mode."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from ..core.random import key_from_seed
from ..core3d.measurements import transition_counts_3d
from ..core3d.random import random_grid_3d, random_rule_3d
from ..core3d.rules import Rule3D, format_rule_3d, parse_rule_3d
from ..core3d.simulate import (
    run_steps_3d,
    run_steps_dynamic_with_metrics_3d,
    run_steps_with_metrics_3d,
    run_until_stable_with_metrics_3d,
)


@dataclass
class SimulationSession3D:
    """Mutable browser state around an otherwise pure JAX 3D grid."""

    grid: jax.Array
    initial_grid: jax.Array
    rule: Rule3D
    generation: int = 0
    seed: int | None = 42
    density: float = 0.04
    speed: float = 10.0
    running: bool = False
    changed_cells: int = 0
    births: int = 0
    deaths: int = 0
    last_step_ms: float = 0.0

    @property
    def depth(self) -> int:
        return int(self.grid.shape[0])

    @property
    def height(self) -> int:
        return int(self.grid.shape[1])

    @property
    def width(self) -> int:
        return int(self.grid.shape[2])

    @property
    def total_cells(self) -> int:
        return self.depth * self.height * self.width


class SessionStore3D:
    """A deliberately small in-memory store for one local server process."""

    def __init__(self) -> None:
        self._sessions: dict[str, SimulationSession3D] = {}

    def create(
        self,
        *,
        depth: int,
        height: int,
        width: int,
        density: float,
        seed: int | None,
        rule: Rule3D | str,
        session_id: str | None = None,
    ) -> str:
        """Create or replace a deterministic 3D session."""

        parsed_rule = parse_rule_3d(rule) if isinstance(rule, str) else rule
        actual_seed = 0 if seed is None else seed
        grid = random_grid_3d(key_from_seed(actual_seed), depth, height, width, density)
        identifier = session_id or uuid4().hex
        self._sessions[identifier] = SimulationSession3D(
            grid=grid,
            initial_grid=grid,
            rule=parsed_rule,
            seed=seed,
            density=float(density),
        )
        return identifier

    def ensure_default(self) -> str:
        """Create the modest, paused default 3D world on first access."""

        if "3d-default" not in self._sessions:
            self.create(
                depth=32,
                height=32,
                width=32,
                density=0.04,
                seed=42,
                rule="B6/S5,6,7",
                session_id="3d-default",
            )
        return "3d-default"

    def get(self, session_id: str = "3d-default") -> SimulationSession3D:
        if session_id == "3d-default" and session_id not in self._sessions:
            self.ensure_default()
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown 3D session: {session_id}") from exc

    @staticmethod
    def _clear_metrics(session: SimulationSession3D) -> None:
        session.changed_cells = 0
        session.births = 0
        session.deaths = 0

    def reset(self, session_id: str = "3d-default") -> SimulationSession3D:
        session = self.get(session_id)
        session.grid = session.initial_grid
        session.generation = 0
        session.running = False
        self._clear_metrics(session)
        return session

    def clear(self, session_id: str = "3d-default") -> SimulationSession3D:
        session = self.get(session_id)
        session.grid = jnp.zeros_like(session.grid)
        session.generation = 0
        session.running = False
        self._clear_metrics(session)
        return session

    def randomize(
        self,
        session_id: str = "3d-default",
        *,
        seed: int | None = 42,
        density: float = 0.04,
    ) -> SimulationSession3D:
        session = self.get(session_id)
        actual_seed = 0 if seed is None else seed
        grid = random_grid_3d(
            key_from_seed(actual_seed),
            session.depth,
            session.height,
            session.width,
            density,
        )
        session.grid = grid
        session.initial_grid = grid
        session.seed = seed
        session.density = float(density)
        session.generation = 0
        session.running = False
        self._clear_metrics(session)
        return session

    def set_rule(self, rule: Rule3D | str, session_id: str = "3d-default") -> SimulationSession3D:
        session = self.get(session_id)
        session.rule = parse_rule_3d(rule) if isinstance(rule, str) else rule
        return session

    def random_rule(
        self, session_id: str = "3d-default", *, seed: int | None = None
    ) -> SimulationSession3D:
        session = self.get(session_id)
        actual_seed = session.seed if seed is None and session.seed is not None else seed
        session.rule = random_rule_3d(key_from_seed(0 if actual_seed is None else actual_seed))
        return session

    def step(
        self,
        session_id: str = "3d-default",
        *,
        steps: int = 1,
        collect_metrics: bool = False,
    ) -> list[dict[str, int | float]]:
        """Advance the complete 3D grid and optionally return compact metrics."""

        session = self.get(session_id)
        start_generation = session.generation
        start_time = perf_counter()
        if collect_metrics:
            final_grid, metric_values = run_steps_with_metrics_3d(session.grid, session.rule, steps)
            host_metrics = np.asarray(jax.device_get(metric_values))
            session.grid = final_grid
            metrics = [
                self._metric_payload(
                    row,
                    generation=start_generation + index + 1,
                    total_cells=session.total_cells,
                )
                for index, row in enumerate(host_metrics)
            ]
            if metrics:
                last = metrics[-1]
                session.changed_cells = int(last["changed_cells"])
                session.births = int(last["births"])
                session.deaths = int(last["deaths"])
        elif steps != 1:
            final_grid, transition = run_steps_dynamic_with_metrics_3d(
                session.grid,
                session.rule,
                steps,
            )
            session.grid = final_grid
            host_transition = np.asarray(jax.device_get(transition))
            session.changed_cells = int(host_transition[1])
            session.births = int(host_transition[2])
            session.deaths = int(host_transition[3])
            metrics = []
        else:
            previous_grid = session.grid
            session.grid = run_steps_3d(previous_grid, session.rule, steps)
            transition = np.asarray(
                jax.device_get(transition_counts_3d(previous_grid, session.grid))
            )
            session.changed_cells = int(transition[1])
            session.births = int(transition[2])
            session.deaths = int(transition[3])
            metrics = []
        session.generation += int(steps)
        session.last_step_ms = (perf_counter() - start_time) * 1000
        return metrics

    def set_speed(self, speed: float, session_id: str = "3d-default") -> SimulationSession3D:
        session = self.get(session_id)
        session.speed = float(speed)
        return session

    def run_until_stable(
        self,
        session_id: str = "3d-default",
        *,
        max_steps: int = 1_000,
    ) -> tuple[int, bool]:
        """Advance until a fixed point or safety bound."""

        session = self.get(session_id)
        final_grid, steps_taken, settled, last_metrics = run_until_stable_with_metrics_3d(
            session.grid,
            session.rule,
            max_steps,
        )
        session.grid = final_grid
        session.generation += int(steps_taken)
        session.running = False
        host_metrics = np.asarray(jax.device_get(last_metrics))
        session.changed_cells = int(host_metrics[1])
        session.births = int(host_metrics[2])
        session.deaths = int(host_metrics[3])
        return int(steps_taken), bool(settled)

    @staticmethod
    def _metric_payload(
        values: Any,
        *,
        generation: int,
        total_cells: int,
    ) -> dict[str, int | float]:
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

    def payload(
        self, session_id: str = "3d-default", *, max_voxels: int = 75_000
    ) -> dict[str, Any]:
        """Return metadata and living coordinates for efficient browser rendering."""

        if max_voxels < 1:
            raise ValueError("max_voxels must be positive")
        session = self.get(session_id)
        values = np.asarray(jax.device_get(session.grid), dtype=np.uint8)
        flat_indices = np.flatnonzero(values)
        render_indices = flat_indices
        sampled = len(flat_indices) > max_voxels
        if sampled:
            render_indices = np.linspace(0, len(flat_indices) - 1, max_voxels, dtype=np.int64)
            render_indices = flat_indices[render_indices]
        coordinates = np.column_stack(np.unravel_index(render_indices, values.shape)).tolist()
        alive = int(values.sum())
        return {
            "session_id": session_id,
            "dimensions": 3,
            "rule": format_rule_3d(session.rule),
            "depth": session.depth,
            "height": session.height,
            "width": session.width,
            "grid_shape": [session.depth, session.height, session.width],
            "seed": session.seed,
            "density": session.density,
            "generation": session.generation,
            "alive": alive,
            "alive_fraction": alive / session.total_cells,
            "changed_cells": session.changed_cells,
            "changed_fraction": session.changed_cells / session.total_cells,
            "births": session.births,
            "birth_fraction": session.births / session.total_cells,
            "deaths": session.deaths,
            "death_fraction": session.deaths / session.total_cells,
            "speed": session.speed,
            "running": session.running,
            "jax_device": str(jax.devices()[0]),
            "last_step_ms": session.last_step_ms,
            "voxels": coordinates,
            "rendered_voxels": len(coordinates),
            "render_sampled": sampled,
            "render_limit": max_voxels,
        }

    def slice_payload(
        self,
        session_id: str = "3d-default",
        *,
        axis: str = "z",
        index: int = 0,
    ) -> dict[str, Any]:
        """Return one exact 2D cross-section for an adjacent canvas."""

        session = self.get(session_id)
        values = np.asarray(jax.device_get(session.grid), dtype=np.uint8)
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        limits = {"z": session.depth, "y": session.height, "x": session.width}
        if index < 0 or index >= limits[axis]:
            raise ValueError(f"slice index must be between 0 and {limits[axis] - 1}")
        selected = {"z": values[index, :, :], "y": values[:, index, :], "x": values[:, :, index]}[
            axis
        ]
        return {
            "session_id": session_id,
            "axis": axis,
            "index": index,
            "height": int(selected.shape[0]),
            "width": int(selected.shape[1]),
            "alive": int(selected.sum()),
            "grid": selected.tolist(),
        }
