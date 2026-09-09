"""HTTP routes for 2D sessions and packed browser rendering."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse, Response

from ..experiments.workload import validate_interactive_workload
from .models import (
    ActionRequest,
    RandomizeRequest,
    RandomRuleRequest,
    RuleRequest,
    SessionRequest,
    StabilityRequest,
    StateRequest,
    StepRequest,
)
from .routes_common import bad_request, session_not_found
from .sessions import SessionStore


def create_router(store: SessionStore) -> APIRouter:
    router = APIRouter(prefix="/api")

    def identifier(payload: ActionRequest | None) -> str:
        return "default" if payload is None else payload.session_id

    @router.post("/session")
    def create_session(payload: SessionRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or SessionRequest()
        try:
            session_id = store.create(
                width=request.width,
                height=request.height,
                density=request.density,
                seed=request.seed,
                rule=request.rule,
                session_id=request.session_id or "default",
            )
            return store.payload(session_id)
        except (TypeError, ValueError, KeyError) as exc:
            raise bad_request(exc) from exc

    @router.get("/state")
    def state(session_id: str = Query(default="default")) -> dict[str, Any]:
        try:
            return store.payload(session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc

    @router.get("/render")
    def render(session_id: str = Query(default="default")) -> Response:
        try:
            content, width, height, generation = store.render_bytes(session_id)
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Grid-Encoding": "packbits-little",
                    "X-Grid-Width": str(width),
                    "X-Grid-Height": str(height),
                    "X-Grid-Generation": str(generation),
                },
            )
        except KeyError as exc:
            raise session_not_found(exc) from exc

    @router.post("/reset")
    def reset(payload: ActionRequest | None = Body(default=None)) -> dict[str, Any]:
        session_id = identifier(payload)
        try:
            store.reset(session_id)
            return store.payload(session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc

    @router.post("/clear")
    def clear(payload: ActionRequest | None = Body(default=None)) -> dict[str, Any]:
        session_id = identifier(payload)
        try:
            store.clear(session_id)
            return store.payload(session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc

    @router.post("/randomize")
    def randomize(payload: RandomizeRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomizeRequest()
        try:
            store.randomize(request.session_id, seed=request.seed, density=request.density)
            return store.payload(request.session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/step")
    def step(payload: StepRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or StepRequest()
        try:
            session = store.get(request.session_id)
            validate_interactive_workload(
                dimensions=2,
                total_cells=session.height * session.width,
                steps=request.steps,
            )
            metrics = store.step(
                request.session_id,
                steps=request.steps,
                collect_metrics=request.collect_metrics,
            )
            response = store.payload(request.session_id)
            if metrics:
                response["metrics"] = metrics
            return response
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/rule")
    def rule(payload: RuleRequest) -> dict[str, Any]:
        try:
            store.set_rule(payload.rule, payload.session_id)
            return store.payload(payload.session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/run-until-stable")
    def run_until_stable(payload: StabilityRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or StabilityRequest()
        try:
            session = store.get(request.session_id)
            validate_interactive_workload(
                dimensions=2,
                total_cells=session.height * session.width,
                steps=request.max_steps,
            )
            steps_run, settled = store.run_until_stable(
                request.session_id,
                max_steps=request.max_steps,
            )
            response = store.payload(request.session_id)
            response["settled"] = settled
            response["steps_run"] = steps_run
            return response
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/random-rule")
    def random_rule(payload: RandomRuleRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomRuleRequest()
        try:
            store.random_rule(request.session_id, seed=request.seed)
            return store.payload(request.session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.put("/state")
    def update_state(payload: StateRequest) -> dict[str, Any]:
        try:
            store.set_state(
                payload.grid,
                payload.session_id,
                rule=payload.rule,
                seed=payload.seed,
                density=payload.density,
                generation=payload.generation,
            )
            return store.payload(payload.session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.get("/export")
    def export_state(session_id: str = Query(default="default")) -> JSONResponse:
        try:
            return JSONResponse(store.payload(session_id, include_grid=True))
        except KeyError as exc:
            raise session_not_found(exc) from exc

    return router
