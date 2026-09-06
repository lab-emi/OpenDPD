"""Local access boundary: loopback-only sessions, CSRF, Host/Origin checks.

Threat model (docs/architecture/threat-model.md): the server listens on
127.0.0.1, but any web page the user visits can still send requests to
``http://127.0.0.1:<port>``. Defences, in order:

1. A one-time **bootstrap token** (printed by the launcher, embedded in the
   URL it opens) is exchanged for a session cookie; API routes require it.
2. State-changing requests must carry ``X-OpenDPD-CSRF`` equal to the
   session's CSRF token. A foreign page cannot read that token and cannot
   add custom headers cross-origin without a CORS preflight, which is
   refused because no CORS headers are ever sent.
3. ``Host`` must be a loopback host and ``Origin``/``Referer`` (when present)
   must match the server's own origin.
4. Request bodies are capped.
"""

from __future__ import annotations

import hmac
import secrets
import threading
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlsplit

SESSION_COOKIE = "opendpd_session"
CSRF_HEADER = "x-opendpd-csrf"
SESSION_MAX_AGE = 7 * 24 * 3600
DEFAULT_MAX_BODY = 2 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@dataclass
class Session:
    session_id: str
    csrf_token: str


class SessionStore:
    """In-memory sessions bound to one server process."""

    def __init__(self, bootstrap_token: Optional[str] = None):
        self.bootstrap_token = bootstrap_token or secrets.token_urlsafe(32)
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def exchange(self, token: str) -> Optional[Session]:
        if not token or not hmac.compare_digest(token, self.bootstrap_token):
            return None
        session = Session(session_id=secrets.token_urlsafe(32), csrf_token=secrets.token_urlsafe(32))
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: Optional[str]) -> Optional[Session]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)


def host_is_loopback(host_header: Optional[str]) -> bool:
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith("["):
        host = host.split("]")[0] + "]"
    else:
        host = host.split(":")[0]
    return host in LOOPBACK_HOSTS


def origin_matches(origin: Optional[str], host_header: Optional[str]) -> bool:
    """True when ``origin`` (an Origin or Referer value) targets this server."""
    if not origin:
        return True   # non-browser clients; CSRF header still required for writes
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    return parts.netloc.lower() == (host_header or "").strip().lower()


class LocalBoundaryMiddleware:
    """Pure ASGI middleware: Host/Origin checks and body size cap for every request."""

    def __init__(self, app, max_body: int = DEFAULT_MAX_BODY):
        self.app = app
        self.max_body = max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        method = scope.get("method", "GET").upper()
        host = headers.get("host")
        if not host_is_loopback(host):
            await _reject(send, 400, "host_not_allowed", "this server only answers loopback hosts")
            return
        if method not in SAFE_METHODS:
            origin = headers.get("origin") or headers.get("referer")
            if not origin_matches(origin, host):
                await _reject(send, 403, "cross_origin_write", "cross-origin state changes are refused")
                return
            length = headers.get("content-length")
            if length and length.isdigit() and int(length) > self.max_body:
                await _reject(send, 413, "payload_too_large", f"request body exceeds {self.max_body} bytes")
                return
        if method == "OPTIONS":
            await _reject(send, 403, "cors_not_supported", "cross-origin requests are not supported")
            return
        await self.app(scope, receive, send)


async def _reject(send, status: int, code: str, message: str) -> None:
    import json
    body = json.dumps({"error": {"code": code, "message": message, "details": [], "hint": None}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
