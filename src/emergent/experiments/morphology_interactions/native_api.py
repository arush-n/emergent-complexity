"""Optional adapter for importing exact grids from the native HTTP simulator."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import numpy as np


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[bytes, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(), response.headers


def load_native_grid(
    base_url: str,
    *,
    width: int,
    height: int,
    density: float,
    seed: int,
    rule: str = "B3/S23",
) -> np.ndarray:
    """Create a native session and decode its exact little-endian render grid.

    The experimental hot loop remains local.  This adapter is only a setup
    bridge/validation path, and uses the server's documented ``/api/session``
    and bit-packed ``/api/render`` endpoints.
    """

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if not 0.0 <= float(density) <= 1.0:
        raise ValueError("density must be between 0 and 1")
    root = base_url.rstrip("/")
    body, _ = _request(
        f"{root}/api/session",
        method="POST",
        payload={
            "width": width,
            "height": height,
            "density": float(density),
            "seed": int(seed),
            "rule": rule,
        },
    )
    try:
        session_payload = json.loads(body.decode("utf-8"))
        session_id = str(session_payload["session_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("native session response did not contain a session_id") from exc

    query = urllib.parse.urlencode({"session_id": session_id})
    packed, headers = _request(f"{root}/api/render?{query}")
    response_width = int(headers.get("X-Grid-Width", width))
    response_height = int(headers.get("X-Grid-Height", height))
    if (response_width, response_height) != (width, height):
        raise ValueError(
            "native render dimensions do not match the requested grid: "
            f"got {(response_width, response_height)}, expected {(width, height)}"
        )
    encoding = headers.get("X-Grid-Encoding", "")
    if encoding.lower() != "packbits-little":
        raise ValueError(f"unsupported native grid encoding: {encoding!r}")
    expected_bytes = (width * height + 7) // 8
    if len(packed) != expected_bytes:
        raise ValueError(
            f"native render has {len(packed)} bytes; expected {expected_bytes} for the grid"
        )
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    return bits[: width * height].reshape((height, width)).astype(np.uint8)


fetch_native_grid = load_native_grid
