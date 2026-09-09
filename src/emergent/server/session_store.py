"""Shared lifecycle bookkeeping for the dimension-specific session stores."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from time import monotonic


def resolve_seed(seed: int | None) -> int:
    """Return the actual seed used for a session.

    ``None`` means "choose a new seed" at the API boundary.  The concrete
    integer is retained in the session so a response can always reproduce the
    state that was generated.
    """

    return int(seed) if seed is not None else secrets.randbits(32)


class SessionLifecycle:
    """Bounded TTL/LRU bookkeeping shared without sharing simulation logic."""

    def _init_lifecycle(
        self,
        *,
        max_sessions: int,
        session_ttl_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        self._last_access: dict[str, float] = {}
        self._max_sessions = max_sessions
        self._session_ttl_seconds = float(session_ttl_seconds)
        self._clock = clock

    @property
    def session_count(self) -> int:
        """Return the number of sessions currently retained by the store."""

        self._prune()
        return len(self._sessions)

    def _remove(self, identifier: str) -> None:
        self._sessions.pop(identifier, None)
        self._last_access.pop(identifier, None)

    def _prune(self, *, protected_id: str | None = None) -> None:
        now = self._clock()
        expired = [
            identifier
            for identifier, last_access in self._last_access.items()
            if now - last_access >= self._session_ttl_seconds
        ]
        for identifier in expired:
            self._remove(identifier)

        overflow = len(self._sessions) - self._max_sessions
        if overflow <= 0:
            return
        candidates = [
            identifier
            for identifier in self._sessions
            if identifier != protected_id
        ]
        candidates.sort(key=lambda identifier: self._last_access.get(identifier, 0.0))
        for identifier in candidates[:overflow]:
            self._remove(identifier)

    def _touch(self, identifier: str) -> None:
        self._last_access[identifier] = self._clock()
