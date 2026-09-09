"""Small helpers shared by the HTTP route modules."""

from __future__ import annotations

from fastapi import HTTPException


def bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def session_not_found(exc: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))
