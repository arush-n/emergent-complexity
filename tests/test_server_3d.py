from fastapi.testclient import TestClient

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
    assert state["voxels"] == []

    randomised = client.post(
        "/api/3d/randomize",
        json={"session_id": "test-3d", "seed": 7, "density": 1},
    )
    assert randomised.status_code == 200
    assert randomised.json()["alive"] == 512
    assert randomised.json()["render_sampled"] is False
    sampled = client.get(
        "/api/3d/state",
        params={"session_id": "test-3d", "max_voxels": 10},
    )
    assert sampled.status_code == 200
    assert sampled.json()["alive"] == 512
    assert sampled.json()["rendered_voxels"] == 10
    assert sampled.json()["render_sampled"] is True

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
            "output_dir": str(tmp_path),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    assert body["manifest"]["dimensions"] == 3
