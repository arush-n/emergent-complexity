from fastapi.testclient import TestClient

from emergent.server.app import create_app
from emergent.server.sessions import SessionStore


def test_api_session_lifecycle() -> None:
    client = TestClient(create_app(SessionStore()))
    created = client.post(
        "/api/session",
        json={
            "session_id": "test",
            "width": 8,
            "height": 8,
            "density": 0,
            "seed": 42,
            "rule": "B3/S23",
        },
    )
    assert created.status_code == 200
    state = created.json()
    assert state["session_id"] == "test"
    assert state["alive"] == 0
    assert state["changed_cells"] == 0
    assert "grid" not in state
    assert "speed" not in state
    assert "running" not in state

    drawn = [[0] * 8 for _ in range(8)]
    drawn[3][3] = 1
    updated = client.put("/api/state", json={"session_id": "test", "grid": drawn})
    assert updated.status_code == 200
    assert updated.json()["alive"] == 1
    assert updated.json()["changed_cells"] == 0

    stepped = client.post(
        "/api/step",
        json={"session_id": "test", "collect_metrics": True},
    )
    assert stepped.status_code == 200
    assert stepped.json()["generation"] == 1
    assert stepped.json()["changed_cells"] == 1
    assert stepped.json()["changed_fraction"] == 1 / 64
    assert stepped.json()["births"] == 0
    assert stepped.json()["deaths"] == 1
    assert stepped.json()["metrics"][0]["generation"] == 1
    assert stepped.json()["metrics"][0]["deaths"] == 1

    batched = client.post(
        "/api/step",
        json={"session_id": "test", "steps": 2, "collect_metrics": True},
    )
    assert batched.status_code == 200
    assert batched.json()["generation"] == 3
    assert [item["generation"] for item in batched.json()["metrics"]] == [2, 3]

    fast = client.post(
        "/api/step",
        json={"session_id": "test", "steps": 25},
    )
    assert fast.status_code == 200
    assert fast.json()["generation"] == 28
    assert "metrics" not in fast.json()

    changed_rule = client.post("/api/rule", json={"session_id": "test", "rule": "B36/S23"})
    assert changed_rule.json()["rule"] == "B36/S23"

    cleared = client.post("/api/clear", json={"session_id": "test"})
    assert cleared.json()["alive"] == 0
    reset = client.post("/api/reset", json={"session_id": "test"})
    assert reset.json()["alive"] == 1

    randomised = client.post(
        "/api/randomize",
        json={"session_id": "test", "seed": 42, "density": 1},
    )
    assert randomised.json()["alive"] == 64

    stabilized = client.post(
        "/api/run-until-stable",
        json={"session_id": "test", "max_steps": 10},
    )
    assert stabilized.status_code == 200
    assert stabilized.json()["settled"] is True
    assert stabilized.json()["steps_run"] == 2
    assert stabilized.json()["alive"] == 0


def test_api_2d_render_is_little_endian_bitpacked() -> None:
    client = TestClient(create_app(SessionStore()))
    created = client.post(
        "/api/session",
        json={"session_id": "packed", "width": 8, "height": 8, "density": 0, "seed": 42},
    )
    assert created.status_code == 200
    drawn = [[0] * 8 for _ in range(8)]
    drawn[0][0] = 1
    drawn[1][3] = 1
    client.put("/api/state", json={"session_id": "packed", "grid": drawn})
    rendered = client.get("/api/render", params={"session_id": "packed"})
    assert rendered.status_code == 200
    assert rendered.headers["x-grid-encoding"] == "packbits-little"
    assert len(rendered.content) == 8
    assert rendered.content[0] & 1
    assert rendered.content[1] & (1 << 3)


def test_api_rejects_oversized_step_and_stability_requests() -> None:
    client = TestClient(create_app(SessionStore()))
    created = client.post(
        "/api/session",
        json={"session_id": "large", "width": 128, "height": 128, "density": 0},
    )
    assert created.status_code == 200

    step = client.post("/api/step", json={"session_id": "large", "steps": 100_000})
    assert step.status_code == 400
    assert "interactive server" in step.json()["detail"]

    stability = client.post(
        "/api/run-until-stable",
        json={"session_id": "large", "max_steps": 100_000},
    )
    assert stability.status_code == 400
    assert "interactive server" in stability.json()["detail"]


def test_api_does_not_enable_wildcard_cors() -> None:
    client = TestClient(create_app(SessionStore()))
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://example.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_api_resolves_null_seed_to_actual_reproducible_integer() -> None:
    client = TestClient(create_app(SessionStore()))
    response = client.post(
        "/api/session",
        json={"session_id": "random-seed", "width": 8, "height": 8, "density": 0, "seed": None},
    )
    assert response.status_code == 200
    assert isinstance(response.json()["seed"], int)


def test_api_2d_experiment_computes_without_writing_files(tmp_path) -> None:
    client = TestClient(create_app(SessionStore()))
    response = client.post(
        "/api/experiments/2d",
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
    assert "output_dir" not in body
    assert list(tmp_path.iterdir()) == []
