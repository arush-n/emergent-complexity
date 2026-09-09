"""Synchronous browser experiment routes with a single-process concurrency guard."""

from __future__ import annotations

from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException

from ..experiments.random_2d import compute_random_2d_experiment
from ..experiments.random_3d import compute_random_3d_experiment
from ..experiments.workload import (
    BROWSER_MAX_CELL_UPDATES,
    INTERACTIVE_MAX_CELL_UPDATES,
    validate_browser_workload,
)
from .models import Experiment2DRequest, Experiment3DRequest


def create_router(experiment_lock: Lock | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/experiments")
    lock = experiment_lock or Lock()

    def acquire() -> None:
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="An experiment is already running. Try again shortly.",
            )

    @router.post("/3d")
    def experiment_3d(payload: Experiment3DRequest) -> dict[str, Any]:
        acquire()
        try:
            validate_browser_workload(
                dimensions=3,
                rules=payload.rules,
                initial_conditions=payload.initial_conditions,
                size=payload.size,
                steps=payload.steps,
            )
            return compute_random_3d_experiment(
                rules=payload.rules,
                initial_conditions=payload.initial_conditions,
                size=payload.size,
                steps=payload.steps,
                density=payload.density,
                seed=payload.seed,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            lock.release()

    @router.post("/2d")
    def experiment_2d(payload: Experiment2DRequest) -> dict[str, Any]:
        acquire()
        try:
            validate_browser_workload(
                dimensions=2,
                rules=payload.rules,
                initial_conditions=payload.initial_conditions,
                size=payload.size,
                steps=payload.steps,
            )
            return compute_random_2d_experiment(
                rules=payload.rules,
                initial_conditions=payload.initial_conditions,
                size=payload.size,
                steps=payload.steps,
                density=payload.density,
                seed=payload.seed,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            lock.release()

    @router.get("/limits")
    def limits() -> dict[str, int]:
        return {
            "browser_max_cell_updates": BROWSER_MAX_CELL_UPDATES,
            "interactive_max_cell_updates": INTERACTIVE_MAX_CELL_UPDATES,
        }

    return router
