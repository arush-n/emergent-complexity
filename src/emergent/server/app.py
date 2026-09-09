"""Application assembly for the JAX cellular-automata laboratory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import jax
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes_2d import create_router as create_2d_router
from .routes_3d import create_router as create_3d_router
from .routes_experiments import create_router as create_experiments_router
from .sessions import SessionStore
from .sessions_3d import SessionStore3D

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
DEFAULT_CORS_ORIGINS = (
    "https://arush-n.github.io",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def cors_origins() -> list[str]:
    """Read explicit browser origins without ever enabling wildcard CORS."""

    configured = os.getenv("EMERGENT_CORS_ORIGINS")
    if configured is None:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(
    store: SessionStore | None = None,
    store_3d: SessionStore3D | None = None,
) -> FastAPI:
    """Create the API application with dimension-specific route modules."""

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
    origins = cors_origins()
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "PUT", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "jax_device": str(jax.devices()[0])}

    application.include_router(create_2d_router(session_store))
    application.include_router(create_3d_router(session_store_3d))
    application.include_router(create_experiments_router(Lock()))

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
