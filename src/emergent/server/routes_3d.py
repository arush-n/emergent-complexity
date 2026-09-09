"""HTTP routes for 3D sessions, packed voxel rendering, and slices."""

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
    Session3DRequest,
    StabilityRequest,
    StepRequest,
)
from .routes_common import bad_request, session_not_found
from .sessions_3d import SessionStore3D


def create_router(store: SessionStore3D) -> APIRouter:
    router = APIRouter(prefix="/api/3d")

    def identifier(payload: ActionRequest | None) -> str:
        return "3d-default" if payload is None else payload.session_id

    @router.post("/session")
    def create_session(payload: Session3DRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or Session3DRequest()
        try:
            session_id = store.create(
                depth=request.depth,
                height=request.height,
                width=request.width,
                density=request.density,
                seed=request.seed,
                rule=request.rule,
                session_id=request.session_id or "3d-default",
            )
            return store.payload(session_id)
        except (TypeError, ValueError, KeyError) as exc:
            raise bad_request(exc) from exc

    @router.get("/state")
    def state(
        session_id: str = Query(default="3d-default"),
        max_voxels: int = Query(default=75_000, ge=1, le=200_000),
    ) -> dict[str, Any]:
        """Return metadata; ``max_voxels`` remains accepted for old clients."""

        del max_voxels
        try:
            return store.payload(session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc

    @router.get("/render")
    def render(
        session_id: str = Query(default="3d-default"),
        max_voxels: int = Query(default=75_000, ge=1, le=200_000),
    ) -> Response:
        try:
            content, metadata = store.render_bytes(session_id, max_voxels=max_voxels)
            headers = {
                "Cache-Control": "no-store",
                "X-Voxel-Encoding": "uint8-xyz-triples",
                "X-Voxel-Depth": str(metadata["depth"]),
                "X-Voxel-Height": str(metadata["height"]),
                "X-Voxel-Width": str(metadata["width"]),
                "X-Voxel-Generation": str(metadata["generation"]),
                "X-Rendered-Voxels": str(metadata["rendered_voxels"]),
                "X-Render-Sampled": str(metadata["render_sampled"]).lower(),
                "X-Render-Limit": str(metadata["render_limit"]),
                "X-Render-Extract-Ms": str(metadata["render_extract_ms"]),
                "X-Serialization-Ms": str(metadata["serialization_ms"]),
            }
            return Response(content=content, media_type="application/octet-stream", headers=headers)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except ValueError as exc:
            raise bad_request(exc) from exc

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
        request = payload or RandomizeRequest(density=0.04)
        session_id = identifier(payload)
        try:
            store.randomize(session_id, seed=request.seed, density=request.density)
            return store.payload(session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/step")
    def step(payload: StepRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or StepRequest(session_id="3d-default")
        session_id = identifier(payload)
        try:
            session = store.get(session_id)
            validate_interactive_workload(
                dimensions=3,
                total_cells=session.total_cells,
                steps=request.steps,
            )
            metrics = store.step(
                session_id,
                steps=request.steps,
                collect_metrics=request.collect_metrics,
            )
            response = store.payload(session_id)
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

    @router.post("/random-rule")
    def random_rule(payload: RandomRuleRequest | None = Body(default=None)) -> dict[str, Any]:
        request = payload or RandomRuleRequest(session_id="3d-default")
        session_id = identifier(payload)
        try:
            store.random_rule(session_id, seed=request.seed)
            return store.payload(session_id)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/run-until-stable")
    def run_until_stable(
        payload: StabilityRequest | None = Body(default=None),
    ) -> dict[str, Any]:
        request = payload or StabilityRequest(session_id="3d-default")
        session_id = identifier(payload)
        try:
            session = store.get(session_id)
            validate_interactive_workload(
                dimensions=3,
                total_cells=session.total_cells,
                steps=request.max_steps,
            )
            steps_run, settled = store.run_until_stable(
                session_id,
                max_steps=request.max_steps,
            )
            response = store.payload(session_id)
            response["settled"] = settled
            response["steps_run"] = steps_run
            return response
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except (TypeError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.get("/slice")
    def slice_3d(
        session_id: str = Query(default="3d-default"),
        axis: str = Query(default="z"),
        index: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return store.slice_payload(session_id, axis=axis, index=index)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except ValueError as exc:
            raise bad_request(exc) from exc

    @router.get("/export")
    def export_state(session_id: str = Query(default="3d-default")) -> Response:
        try:
            content, generation = store.export_npz(session_id)
            return Response(
                content=content,
                media_type="application/zip",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="life-lab-3d-gen-{generation}.npz"'
                    )
                },
            )
        except KeyError as exc:
            raise session_not_found(exc) from exc

    @router.get("/export-view")
    def export_view(
        session_id: str = Query(default="3d-default"),
        max_voxels: int = Query(default=75_000, ge=1, le=200_000),
    ) -> JSONResponse:
        """Export the capped render view, explicitly distinct from exact state."""

        try:
            payload = store.render_view_payload(session_id, max_voxels=max_voxels)
            payload["export_kind"] = "render_view"
            payload["exact"] = False
            payload["export_note"] = (
                "This JSON contains displayed living voxel coordinates only. "
                "Use the exact NPZ export for the complete grid."
            )
            return JSONResponse(payload)
        except KeyError as exc:
            raise session_not_found(exc) from exc
        except ValueError as exc:
            raise bad_request(exc) from exc

    return router
