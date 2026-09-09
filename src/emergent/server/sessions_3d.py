"""Lightweight in-memory sessions for the 3D browser mode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from ..core.random import key_from_seed
from ..core3d.measurements import alive_count_3d, transition_counts_3d
from ..core3d.random import random_grid_3d, random_rule_3d
from ..core3d.rules import Rule3D, format_rule_3d, parse_rule_3d
from ..core3d.simulate import (
    run_steps_3d,
    run_steps_dynamic_with_metrics_3d,
    run_steps_with_metrics_3d,
    run_until_stable_with_metrics_3d,
)
from .session_store import SessionLifecycle, resolve_seed


@dataclass
class SimulationSession3D:
    """Mutable browser state around an otherwise pure JAX 3D grid."""

    grid: jax.Array
    initial_grid: jax.Array
    rule: Rule3D
    generation: int = 0
    seed: int = 42
    initial_density: float = 0.04
    changed_cells: int = 0
    births: int = 0
    deaths: int = 0
    last_step_ms: float = 0.0
    last_render_extract_ms: float = 0.0
    last_serialization_ms: float = 0.0

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


class SessionStore3D(SessionLifecycle):
    """A bounded, expiring in-memory store for one server process."""

    def __init__(
        self,
        *,
        max_sessions: int = 256,
        session_ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._sessions: dict[str, SimulationSession3D] = {}
        self._init_lifecycle(
            max_sessions=max_sessions,
            session_ttl_seconds=session_ttl_seconds,
            clock=clock,
        )

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
        actual_seed = resolve_seed(seed)
        grid = random_grid_3d(key_from_seed(actual_seed), depth, height, width, density)
        identifier = session_id or uuid4().hex
        self._sessions[identifier] = SimulationSession3D(
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
        """Create the modest, paused default 3D world on first access."""

        self._prune()
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
        self._touch("3d-default")
        return "3d-default"

    def get(self, session_id: str = "3d-default") -> SimulationSession3D:
        self._prune()
        if session_id == "3d-default" and session_id not in self._sessions:
            self.ensure_default()
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown 3D session: {session_id}") from exc
        self._touch(session_id)
        return session

    @staticmethod
    def _clear_metrics(session: SimulationSession3D) -> None:
        session.changed_cells = 0
        session.births = 0
        session.deaths = 0

    def reset(self, session_id: str = "3d-default") -> SimulationSession3D:
        session = self.get(session_id)
        session.grid = session.initial_grid
        session.generation = 0
        self._clear_metrics(session)
        return session

    def clear(self, session_id: str = "3d-default") -> SimulationSession3D:
        session = self.get(session_id)
        session.grid = jnp.zeros_like(session.grid)
        session.generation = 0
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
        actual_seed = resolve_seed(seed)
        grid = random_grid_3d(
            key_from_seed(actual_seed),
            session.depth,
            session.height,
            session.width,
            density,
        )
        session.grid = grid
        session.initial_grid = grid
        session.seed = actual_seed
        session.initial_density = float(density)
        session.generation = 0
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
        actual_seed = session.seed if seed is None else resolve_seed(seed)
        session.rule = random_rule_3d(key_from_seed(actual_seed))
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

    def payload(self, session_id: str = "3d-default") -> dict[str, Any]:
        """Return metadata without serializing the potentially large voxel set."""

        session = self.get(session_id)
        alive = int(alive_count_3d(session.grid))
        alive_fraction = alive / session.total_cells
        response = {
            "session_id": session_id,
            "dimensions": 3,
            "rule": format_rule_3d(session.rule),
            "depth": session.depth,
            "height": session.height,
            "width": session.width,
            "grid_shape": [session.depth, session.height, session.width],
            "seed": session.seed,
            "initial_density": session.initial_density,
            "generation": session.generation,
            "alive": alive,
            "total_cells": session.total_cells,
            "alive_fraction": alive_fraction,
            "changed_cells": session.changed_cells,
            "changed_fraction": session.changed_cells / session.total_cells,
            "births": session.births,
            "birth_fraction": session.births / session.total_cells,
            "deaths": session.deaths,
            "death_fraction": session.deaths / session.total_cells,
            "jax_device": str(jax.devices()[0]),
            "last_step_ms": session.last_step_ms,
            "simulation_ms": session.last_step_ms,
        }
        return response

    def render_bytes(
        self, session_id: str = "3d-default", *, max_voxels: int = 75_000
    ) -> tuple[bytes, dict[str, int | bool | float]]:
        """Return sampled living voxel coordinates as packed uint8 triples."""

        if max_voxels < 1:
            raise ValueError("max_voxels must be positive")
        session = self.get(session_id)
        extract_start = perf_counter()
        values = np.asarray(jax.device_get(session.grid), dtype=np.uint8)
        flat_indices = np.flatnonzero(values)
        sampled = len(flat_indices) > max_voxels
        if sampled:
            render_indices = np.linspace(0, len(flat_indices) - 1, max_voxels, dtype=np.int64)
            render_indices = flat_indices[render_indices]
        else:
            render_indices = flat_indices
        coordinates = np.column_stack(np.unravel_index(render_indices, values.shape)).astype(
            np.uint8, copy=False
        )
        render_extract_ms = (perf_counter() - extract_start) * 1000
        serialization_start = perf_counter()
        content = coordinates.tobytes()
        serialization_ms = (perf_counter() - serialization_start) * 1000
        session.last_render_extract_ms = render_extract_ms
        session.last_serialization_ms = serialization_ms
        metadata: dict[str, int | bool | float] = {
            "depth": session.depth,
            "height": session.height,
            "width": session.width,
            "generation": session.generation,
            "rendered_voxels": len(coordinates),
            "render_sampled": sampled,
            "render_limit": max_voxels,
            "render_extract_ms": render_extract_ms,
            "serialization_ms": serialization_ms,
        }
        return content, metadata

    def render_view_payload(
        self, session_id: str = "3d-default", *, max_voxels: int = 75_000
    ) -> dict[str, Any]:
        """Return the capped coordinate view for explicit JSON export only."""

        content, metadata = self.render_bytes(session_id, max_voxels=max_voxels)
        coordinates = np.frombuffer(content, dtype=np.uint8).reshape(-1, 3).tolist()
        response = self.payload(session_id)
        response.update(metadata)
        response["voxels"] = coordinates
        return response

    def export_npz(self, session_id: str = "3d-default") -> tuple[bytes, int]:
        """Return an exact NPZ state and its generation for a browser download."""

        from ..io.serialization_3d import state_to_npz_bytes_3d

        session = self.get(session_id)
        content = state_to_npz_bytes_3d(
            session.grid,
            session.rule,
            seed=session.seed,
            density=session.initial_density,
            generation=session.generation,
        )
        return content, session.generation

    def slice_payload(
        self,
        session_id: str = "3d-default",
        *,
        axis: str = "z",
        index: int = 0,
    ) -> dict[str, Any]:
        """Return one exact 2D cross-section for an adjacent canvas."""

        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        session = self.get(session_id)
        limits = {"z": session.depth, "y": session.height, "x": session.width}
        if index < 0 or index >= limits[axis]:
            raise ValueError(f"slice index must be between 0 and {limits[axis] - 1}")
        if axis == "z":
            selected = session.grid[index, :, :]
        elif axis == "y":
            selected = session.grid[:, index, :]
        else:
            selected = session.grid[:, :, index]
        extract_start = perf_counter()
        selected_host = np.asarray(jax.device_get(selected), dtype=np.uint8)
        render_extract_ms = (perf_counter() - extract_start) * 1000
        serialization_start = perf_counter()
        response = {
            "session_id": session_id,
            "axis": axis,
            "index": index,
            "height": int(selected_host.shape[0]),
            "width": int(selected_host.shape[1]),
            "alive": int(selected_host.sum()),
            "grid": selected_host.tolist(),
            "render_extract_ms": render_extract_ms,
            "serialization_ms": 0.0,
        }
        response["serialization_ms"] = (perf_counter() - serialization_start) * 1000
        return response
