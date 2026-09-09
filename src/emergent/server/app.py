"""FastAPI adapter exposing the JAX simulator to the local canvas UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import jax
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    ActionRequest,
    Experiment2DRequest,
    Experiment3DRequest,
    RandomizeRequest,
    RandomRuleRequest,
    RuleRequest,
    Session3DRequest,
    SessionRequest,
    SpeedRequest,
    StabilityRequest,
    StateRequest,
    StepRequest,
)
from .sessions import SessionStore
from .sessions_3d import SessionStore3D

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


def _session_id(payload: ActionRequest | None) -> str:
    return "default" if payload is None else payload.session_id


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def create_app(
    store: SessionStore | None = None,
    store_3d: SessionStore3D | None = None,
) -> FastAPI:
    """Create the API application, optionally with an injected session store."""

    session_store = store or SessionStore()
    session_store_3d = store_3d or SessionStore3D()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("JAX device: %s", jax.devices()[0])
        yield

    application = FastAPI(
        title="Emergent Complexity Cellular Automata Lab",
        version="0.1.0",
        description="A thin HTTP layer over a JAX cellular-automata engine.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "jax_device": str(jax.devices()[0])}

    @application.post("/api/session")
    def create_session(payload: SessionRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or SessionRequest()
        try:
            identifier = session_store.create(
                width=request.width,
                height=request.height,
                density=request.density,
                seed=request.seed,
                rule=request.rule,
                session_id=request.session_id or "default",
            )
            return session_store.payload(identifier)
        except (TypeError, ValueError, KeyError) as exc:
            raise _bad_request(exc) from exc

    @application.get("/api/state")
    def state(session_id: str = Query(default="default")) -> dict[str, Any]:
        try:
            return session_store.payload(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/reset")
    def reset(payload: ActionRequest | None = Body(default=None)) -> dict[str, Any]:
        try:
            identifier = _session_id(payload)
            session_store.reset(identifier)
            return session_store.payload(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/clear")
    def clear(payload: ActionRequest | None = Body(default=None)) -> dict[str, Any]:
        try:
            identifier = _session_id(payload)
            session_store.clear(identifier)
            return session_store.payload(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/randomize")
    def randomize(payload: RandomizeRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomizeRequest()
        try:
            session_store.randomize(
                request.session_id,
                seed=request.seed,
                density=request.density,
            )
            return session_store.payload(request.session_id)
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise _bad_request(exc) from exc

    @application.post("/api/step")
    def step(payload: StepRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or StepRequest()
        try:
            metrics = session_store.step(
                request.session_id,
                steps=request.steps,
                collect_metrics=request.collect_metrics,
            )
            response = session_store.payload(request.session_id)
            if metrics:
                response["metrics"] = metrics
            return response
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise _bad_request(exc) from exc

    @application.post("/api/rule")
    def rule(payload: RuleRequest) -> dict[str, Any]:
        try:
            session_store.set_rule(payload.rule, payload.session_id)
            return session_store.payload(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise _bad_request(exc) from exc

    @application.post("/api/run-until-stable")
    def run_until_stable(payload: StabilityRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or StabilityRequest()
        try:
            steps_run, settled = session_store.run_until_stable(
                request.session_id,
                max_steps=request.max_steps,
            )
            response = session_store.payload(request.session_id)
            response["settled"] = settled
            response["steps_run"] = steps_run
            return response
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise _bad_request(exc) from exc

    @application.post("/api/random-rule")
    def random_rule(payload: RandomRuleRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomRuleRequest()
        try:
            session_store.random_rule(request.session_id, seed=request.seed)
            return session_store.payload(request.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.put("/api/state")
    def update_state(payload: StateRequest) -> dict[str, Any]:
        try:
            session_store.set_state(
                payload.grid,
                payload.session_id,
                rule=payload.rule,
                seed=payload.seed,
                density=payload.density,
                generation=payload.generation,
            )
            return session_store.payload(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise _bad_request(exc) from exc

    @application.post("/api/speed")
    def speed(payload: SpeedRequest) -> dict[str, Any]:
        try:
            session_store.set_speed(payload.speed, payload.session_id)
            return session_store.payload(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/api/export")
    def export_state(session_id: str = Query(default="default")) -> JSONResponse:
        try:
            return JSONResponse(session_store.payload(session_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/3d/session")
    def create_session_3d(payload: Session3DRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or Session3DRequest()
        try:
            identifier = session_store_3d.create(
                depth=request.depth,
                height=request.height,
                width=request.width,
                density=request.density,
                seed=request.seed,
                rule=request.rule,
                session_id=request.session_id or "3d-default",
            )
            return session_store_3d.payload(identifier)
        except (TypeError, ValueError, KeyError) as exc:
            raise _bad_request(exc) from exc

    @application.get("/api/3d/state")
    def state_3d(
        session_id: str = Query(default="3d-default"),
        max_voxels: int = Query(default=75_000, ge=1, le=200_000),
    ) -> dict[str, Any]:
        try:
            return session_store_3d.payload(session_id, max_voxels=max_voxels)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @application.post("/api/3d/reset")
    def reset_3d(payload: ActionRequest | None = Body(default=None)) -> dict[str, Any]:
        identifier = _session_id(payload) if payload else "3d-default"
        try:
            session_store_3d.reset(identifier)
            return session_store_3d.payload(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/3d/clear")
    def clear_3d(payload: ActionRequest | None = Body(default=None)) -> dict[str, Any]:
        identifier = _session_id(payload) if payload else "3d-default"
        try:
            session_store_3d.clear(identifier)
            return session_store_3d.payload(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/3d/randomize")
    def randomize_3d(payload: RandomizeRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomizeRequest(density=0.04)
        identifier = request.session_id if payload else "3d-default"
        try:
            session_store_3d.randomize(
                identifier,
                seed=request.seed,
                density=request.density,
            )
            return session_store_3d.payload(identifier)
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise _bad_request(exc) from exc

    @application.post("/api/3d/step")
    def step_3d(payload: StepRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or StepRequest()
        identifier = request.session_id if payload else "3d-default"
        try:
            metrics = session_store_3d.step(
                identifier,
                steps=request.steps,
                collect_metrics=request.collect_metrics,
            )
            response = session_store_3d.payload(identifier)
            if metrics:
                response["metrics"] = metrics
            return response
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise _bad_request(exc) from exc

    @application.post("/api/3d/rule")
    def rule_3d(payload: RuleRequest) -> dict[str, Any]:
        try:
            session_store_3d.set_rule(payload.rule, payload.session_id)
            return session_store_3d.payload(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise _bad_request(exc) from exc

    @application.post("/api/3d/random-rule")
    def random_rule_3d(payload: RandomRuleRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomRuleRequest()
        identifier = request.session_id if payload else "3d-default"
        try:
            session_store_3d.random_rule(identifier, seed=request.seed)
            return session_store_3d.payload(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/3d/run-until-stable")
    def run_until_stable_3d(
        payload: StabilityRequest | None = Body(default=None),
    ) -> dict[str, Any]:
        request = payload or StabilityRequest()
        identifier = request.session_id if payload else "3d-default"
        try:
            steps_run, settled = session_store_3d.run_until_stable(
                identifier,
                max_steps=request.max_steps,
            )
            response = session_store_3d.payload(identifier)
            response["settled"] = settled
            response["steps_run"] = steps_run
            return response
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise _bad_request(exc) from exc

    @application.post("/api/3d/speed")
    def speed_3d(payload: SpeedRequest) -> dict[str, Any]:
        try:
            session_store_3d.set_speed(payload.speed, payload.session_id)
            return session_store_3d.payload(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/api/3d/slice")
    def slice_3d(
        session_id: str = Query(default="3d-default"),
        axis: str = Query(default="z"),
        index: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return session_store_3d.slice_payload(session_id, axis=axis, index=index)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @application.get("/api/3d/export")
    def export_state_3d(
        session_id: str = Query(default="3d-default"),
        max_voxels: int = Query(default=75_000, ge=1, le=200_000),
    ) -> JSONResponse:
        try:
            payload = session_store_3d.payload(session_id, max_voxels=max_voxels)
            payload["export_note"] = (
                "Export contains rendered living voxel coordinates; the Python API supports "
                "full NPZ/JSON state persistence."
            )
            return JSONResponse(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/experiments/3d")
    def experiment_3d(payload: Experiment3DRequest) -> dict[str, Any]:
        from ..experiments.random_3d import run_random_3d_experiment

        try:
            return run_random_3d_experiment(
                rules=payload.rules,
                initial_conditions=payload.initial_conditions,
                size=payload.size,
                steps=payload.steps,
                density=payload.density,
                seed=payload.seed,
                output_dir=payload.output_dir,
            )
        except (TypeError, ValueError, OSError) as exc:
            raise _bad_request(exc) from exc

    @application.post("/api/experiments/2d")
    def experiment_2d(payload: Experiment2DRequest) -> dict[str, Any]:
        from ..experiments.random_2d import run_random_2d_experiment

        try:
            return run_random_2d_experiment(
                rules=payload.rules,
                initial_conditions=payload.initial_conditions,
                size=payload.size,
                steps=payload.steps,
                density=payload.density,
                seed=payload.seed,
                output_dir=payload.output_dir,
            )
        except (TypeError, ValueError, OSError) as exc:
            raise _bad_request(exc) from exc

    if FRONTEND_DIR.exists():
        application.mount(
            "/",
            StaticFiles(directory=str(FRONTEND_DIR), html=True),
            name="frontend",
        )
    return application


app = create_app()


def main() -> None:
    """Run the local development server."""

    import uvicorn

    uvicorn.run("emergent.server.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
