import io
import json

import jax.numpy as jnp
import numpy as np
from fastapi.testclient import TestClient

import emergent.server.sessions_3d as sessions_3d_module
from emergent.server.app import create_app
from emergent.server.sessions import SessionStore
from emergent.server.sessions_3d import SessionStore3D


def test_api_3d_session_lifecycle() -> None:
    client = TestClient(create_app(SessionStore(), SessionStore3D()))
    created = client.post(
        "/api/3d/session",
        json={
            "session_id": "test-3d",
            "depth": 8,
            "height": 8,
            "width": 8,
            "density": 0,
            "seed": 42,
            "rule": "B1/S",
        },
    )
    assert created.status_code == 200
    state = created.json()
    assert state["dimensions"] == 3
    assert state["grid_shape"] == [8, 8, 8]
    assert state["alive"] == 0
    assert "voxels" not in state
    assert "speed" not in state
    assert "running" not in state

    rendered = client.get("/api/3d/render", params={"session_id": "test-3d"})
    assert rendered.status_code == 200
    assert rendered.content == b""
    assert rendered.headers["x-voxel-encoding"] == "uint8-xyz-triples"

    randomised = client.post(
        "/api/3d/randomize",
        json={"session_id": "test-3d", "seed": 7, "density": 1},
    )
    assert randomised.status_code == 200
    assert randomised.json()["alive"] == 512
    full_render = client.get("/api/3d/render", params={"session_id": "test-3d"})
    assert full_render.headers["x-render-sampled"] == "false"
    sampled = client.get(
        "/api/3d/render",
        params={"session_id": "test-3d", "max_voxels": 10},
    )
    assert sampled.status_code == 200
    assert len(sampled.content) == 30
    assert sampled.headers["x-rendered-voxels"] == "10"
    assert sampled.headers["x-render-sampled"] == "true"

    changed_rule = client.post(
        "/api/3d/rule",
        json={"session_id": "test-3d", "rule": "B/S2,6"},
    )
    assert changed_rule.json()["rule"] == "B/S2,6"

    stepped = client.post(
        "/api/3d/step",
        json={"session_id": "test-3d", "collect_metrics": True},
    )
    assert stepped.status_code == 200
    assert stepped.json()["generation"] == 1
    assert stepped.json()["metrics"][0]["deaths"] == 512

    slice_response = client.get(
        "/api/3d/slice",
        params={"session_id": "test-3d", "axis": "z", "index": 0},
    )
    assert slice_response.status_code == 200
    assert slice_response.json()["grid"] == [[0] * 8 for _ in range(8)]

    cleared = client.post("/api/3d/clear", json={"session_id": "test-3d"})
    assert cleared.status_code == 200
    assert cleared.json()["alive"] == 0
    reset = client.post("/api/3d/reset", json={"session_id": "test-3d"})
    assert reset.status_code == 200
    assert reset.json()["alive"] == 512


def test_api_3d_experiment_endpoint_returns_raw_rows(tmp_path) -> None:
    client = TestClient(create_app(SessionStore(), SessionStore3D()))
    response = client.post(
        "/api/experiments/3d",
        json={
            "rules": 1,
            "initial_conditions": 1,
            "size": 8,
            "steps": 1,
            "density": 0,
            "seed": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    assert body["manifest"]["dimensions"] == 3
    assert "output_dir" not in body
    assert list(tmp_path.iterdir()) == []


def test_api_experiment_endpoint_rejects_browser_workloads_that_are_too_large() -> None:
    client = TestClient(create_app(SessionStore(), SessionStore3D()))
    response = client.post(
        "/api/experiments/3d",
        json={
            "rules": 100,
            "initial_conditions": 100,
            "size": 96,
            "steps": 2_000,
            "density": 0.1,
            "seed": 42,
        },
    )
    assert response.status_code == 400
    assert "Use the CLI experiment runner" in response.json()["detail"]


def test_api_3d_rejects_oversized_step_and_stability_requests() -> None:
    store_3d = SessionStore3D()
    client = TestClient(create_app(SessionStore(), store_3d))
    created = client.post(
        "/api/3d/session",
        json={
            "session_id": "large-3d",
            "depth": 32,
            "height": 32,
            "width": 32,
            "density": 0,
            "seed": 1,
            "rule": "B1/S",
        },
    )
    assert created.status_code == 200

    step = client.post("/api/3d/step", json={"session_id": "large-3d", "steps": 5_000})
    assert step.status_code == 400
    assert "interactive server" in step.json()["detail"]

    stability = client.post(
        "/api/3d/run-until-stable",
        json={"session_id": "large-3d", "max_steps": 5_000},
    )
    assert stability.status_code == 400
    assert "interactive server" in stability.json()["detail"]


def test_api_3d_slices_are_exact_for_all_axes_and_transfer_only_a_plane(monkeypatch) -> None:
    store = SessionStore3D()
    store.create(
        depth=3,
        height=4,
        width=5,
        density=0,
        seed=1,
        rule="B/S",
        session_id="slice-3d",
    )
    grid = jnp.zeros((3, 4, 5), dtype=jnp.uint8)
    grid = grid.at[1, 2, 3].set(1).at[2, 0, 4].set(1)
    session = store.get("slice-3d")
    session.grid = grid

    original_device_get = sessions_3d_module.jax.device_get
    transferred_shapes = []

    def record_shape(value):
        transferred_shapes.append(tuple(value.shape))
        return original_device_get(value)

    monkeypatch.setattr(sessions_3d_module.jax, "device_get", record_shape)
    client = TestClient(create_app(SessionStore(), store))
    indices = {"z": 1, "y": 2, "x": 4}
    slices = {
        axis: client.get(
            "/api/3d/slice",
            params={"session_id": "slice-3d", "axis": axis, "index": indices[axis]},
        ).json()
        for axis in ("z", "y", "x")
    }

    assert slices["z"]["height"] == 4
    assert slices["z"]["width"] == 5
    assert slices["z"]["alive"] == 1
    assert slices["y"]["height"] == 3
    assert slices["y"]["width"] == 5
    assert slices["y"]["alive"] == 1
    assert slices["x"]["height"] == 3
    assert slices["x"]["width"] == 4
    assert slices["x"]["alive"] == 1
    assert transferred_shapes == [(4, 5), (3, 5), (3, 4)]


def test_api_3d_metadata_and_render_are_separate() -> None:
    store = SessionStore3D()
    store.create(
        depth=3,
        height=4,
        width=5,
        density=0,
        seed=1,
        rule="B/S",
        session_id="payload-3d",
    )
    session = store.get("payload-3d")
    session.grid = session.grid.at[1, 2, 3].set(1)
    payload = store.payload("payload-3d")

    assert payload["alive"] == 1
    assert "voxels" not in payload
    content, metadata = store.render_bytes("payload-3d")
    assert len(content) == 3
    assert metadata["rendered_voxels"] == 1


def test_api_3d_export_is_an_exact_npz_and_view_export_is_explicit() -> None:
    store = SessionStore3D()
    store.create(
        depth=8,
        height=8,
        width=8,
        density=0,
        seed=2,
        rule="B1/S",
        session_id="export-3d",
    )
    session = store.get("export-3d")
    session.grid = session.grid.at[2, 3, 4].set(1)
    session.generation = 7
    client = TestClient(create_app(SessionStore(), store))

    response = client.get("/api/3d/export", params={"session_id": "export-3d"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert response.headers["content-disposition"].endswith('life-lab-3d-gen-7.npz"')
    with np.load(io.BytesIO(response.content), allow_pickle=False) as archive:
        assert archive["grid"].shape == (8, 8, 8)
        assert int(archive["grid"].sum()) == 1
        assert json.loads(str(archive["metadata"].item()))["generation"] == 7

    view = client.get("/api/3d/export-view", params={"session_id": "export-3d", "max_voxels": 1})
    assert view.status_code == 200
    assert view.json()["export_kind"] == "render_view"
    assert view.json()["exact"] is False
