import pytest

from emergent.server.sessions import SessionStore
from emergent.server.sessions_3d import SessionStore3D


def test_2d_sessions_expire_and_are_bounded() -> None:
    now = [0.0]
    store = SessionStore(max_sessions=2, session_ttl_seconds=10, clock=lambda: now[0])
    store.create(width=8, height=8, density=0, seed=1, rule="B3/S23", session_id="one")
    now[0] = 1
    store.create(width=8, height=8, density=0, seed=2, rule="B3/S23", session_id="two")
    now[0] = 2
    store.create(width=8, height=8, density=0, seed=3, rule="B3/S23", session_id="three")
    assert store.session_count == 2
    with pytest.raises(KeyError, match="one"):
        store.get("one")

    now[0] = 13
    with pytest.raises(KeyError, match="two"):
        store.get("two")
    assert store.session_count == 0


def test_3d_sessions_expire_and_are_bounded() -> None:
    now = [0.0]
    store = SessionStore3D(max_sessions=2, session_ttl_seconds=10, clock=lambda: now[0])
    for index, identifier in enumerate(("one", "two", "three")):
        store.create(
            depth=8,
            height=8,
            width=8,
            density=0,
            seed=index,
            rule="B1/S",
            session_id=identifier,
        )
        now[0] += 1
    assert store.session_count == 2
    with pytest.raises(KeyError, match="one"):
        store.get("one")

    now[0] = 13
    with pytest.raises(KeyError, match="two"):
        store.get("two")
    assert store.session_count == 0
