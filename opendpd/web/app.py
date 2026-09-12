"""Public ASGI boundary with isolated local-app instances; no host compute."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse, Response

from opendpd import __version__
from opendpd.server.security import CSRF_HEADER, SESSION_COOKIE
from opendpd.web.policy import WebConfig, allowed, check_body, check_query, reject
from opendpd.web.runtime import TenantManager

log = logging.getLogger(__name__)
PREFIX = "/api/v1"


def session_info(tenant=None, token=None):
    result = {"authenticated": tenant is not None, "csrf_token": None, "version": __version__, "mode": "web"}
    if tenant:
        result["expires_at"] = datetime.fromtimestamp(tenant.expires_at, timezone.utc).isoformat()
    if token:
        result["access_token"] = token
    return result


class PublicBoundary:
    def __init__(self, app, *, manager: TenantManager):
        self.app, self.manager, self.config = app, manager, manager.config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        origin = request.headers.get("origin")
        started = False

        async def secure_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                # The nested local app's same-origin headers do not describe this
                # cross-origin API. Never forward local cookies or cache headers.
                replaced = {b"set-cookie", b"content-security-policy", b"cache-control", b"cross-origin-resource-policy",
                            b"access-control-allow-origin", b"access-control-allow-credentials", b"referrer-policy"}
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in replaced]
                headers.extend([(b"cache-control", b"no-store"), (b"referrer-policy", b"no-referrer"),
                                (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'; sandbox"),
                                (b"x-content-type-options", b"nosniff"), (b"vary", b"Origin"),
                                (b"strict-transport-security", b"max-age=31536000")])
                if origin == self.config.origin:
                    headers.append((b"access-control-allow-origin", origin.encode()))
                    headers.append((b"access-control-expose-headers", b"Content-Disposition, Retry-After"))
                if message["status"] == 429:
                    headers.append((b"retry-after", b"60"))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.handle(request, scope, receive, secure_send)
        except HTTPException as exc:
            if not started:
                await JSONResponse(exc.detail, status_code=exc.status_code)(scope, receive, secure_send)
        except Exception:
            log.exception("public API request failed")
            if not started:
                await JSONResponse({"error": {"code": "service_error", "message": "the service could not complete this request",
                                             "details": [], "hint": None}}, status_code=500)(scope, receive, secure_send)

    async def handle(self, request, scope, receive, send):
        headers = request.headers
        # Uvicorn MUST run with proxy_headers=False. Only QEMU's private host
        # gateway / loopback is trusted; X-Forwarded-For never determines identity.
        peer = request.client.host if request.client else ""
        if peer not in {"127.0.0.1", "::1", "10.0.2.2"}:
            reject(403, "untrusted_peer", "requests must arrive through the configured tunnel")
        if request.url.path == "/healthz" and headers.get("host") in {"127.0.0.1", "localhost"}:
            code = 200 if self.manager.cleanup_healthy else 503
            await JSONResponse({"status": "ok" if code == 200 else "cleanup_failed"}, status_code=code)(scope, receive, send)
            return
        for key in ("host", "origin", "authorization", "cf-connecting-ip", "content-length", "x-forwarded-proto"):
            if len(headers.getlist(key)) > 1:
                reject(400, "ambiguous_headers", "duplicate security headers are refused")
        if headers.get("host") != self.config.tunnel_host or headers.get("x-forwarded-proto") != "https":
            reject(403, "tunnel_required", "requests must arrive through the configured HTTPS tunnel")
        if headers.get("origin") != self.config.origin:
            reject(403, "origin_not_allowed", "this origin is not allowed")
        try:
            address = str(ipaddress.ip_address(headers.get("cf-connecting-ip", "")))
        except ValueError:
            reject(403, "client_ip_required", "a verified Cloudflare client address is required")
        ip_key = self.manager.ip_key(address)
        self.manager.rate_limit(ip_key)
        path = request.url.path
        if not path.startswith(PREFIX + "/") or len(path) > 512 or len(scope.get("query_string", b"")) > 4096:
            reject(404, "not_found", "unknown public API route")
        path = path[len(PREFIX):]
        method = request.method
        special = path in {"/session", "/web/sessions"}
        requested = headers.get("access-control-request-method", "") if method == "OPTIONS" else method
        if not (allowed(requested, path) or (path == "/session" and requested == "GET")
                or (path == "/web/sessions" and requested == "POST")):
            reject(403, "feature_unavailable", "this feature is unavailable in the public demo")
        if method == "OPTIONS":
            requested_headers = {h.strip().lower() for h in headers.get("access-control-request-headers", "").split(",") if h.strip()}
            if not requested_headers <= {"authorization", "content-type", "accept"}:
                reject(403, "headers_not_allowed", "unsupported cross-origin headers")
            await Response(status_code=204, headers={"Access-Control-Allow-Methods": "GET, POST, PUT",
                               "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept",
                               "Access-Control-Max-Age": "600"})(scope, receive, send)
            return
        check_query(request.query_params)
        auth = headers.get("authorization", "")
        tenant = None
        if auth:
            if not auth.startswith("Bearer ") or len(auth) > 128:
                reject(401, "unauthorized", "a valid bearer session is required")
            tenant = self.manager.authenticate(auth[7:])
        if not tenant and not special:
            reject(401, "unauthorized", "start a temporary session first")
        if self.manager.inflight >= self.config.max_requests or (tenant and tenant.inflight >= 3):
            reject(429, "service_busy", "too many simultaneous requests; try again later")
        # Reserve before receiving the body, including chunked and slow requests.
        self.manager.inflight += 1
        if tenant:
            tenant.inflight += 1
        try:
            data = b""
            length = headers.get("content-length")
            if length is not None and (not length.isdecimal() or int(length) > self.config.max_body):
                reject(413, "payload_too_large", "request body exceeds the public API limit")
            if method in {"POST", "PUT"} and headers.get("content-type", "").split(";")[0] != "application/json":
                reject(415, "json_required", "only JSON requests are accepted; uploads are disabled")

            async def read_body():
                chunks = bytearray()
                async for chunk in request.stream():
                    chunks.extend(chunk)
                    if len(chunks) > self.config.max_body:
                        reject(413, "payload_too_large", "request body exceeds the public API limit")
                return bytes(chunks)

            try:
                data = await asyncio.wait_for(read_body(), timeout=10)
                body = json.loads(data) if data else {}
            except (ValueError, RecursionError):
                reject(422, "invalid_json", "invalid JSON body")
            except asyncio.TimeoutError:
                reject(408, "request_timeout", "request body took too long")
            if path == "/web/sessions":
                if body != {}:
                    reject(422, "invalid_request", "session creation takes an empty JSON object")
                if tenant:
                    await JSONResponse(session_info(tenant))(scope, receive, send)
                else:
                    created, token = await self.manager.create(ip_key)
                    await JSONResponse(session_info(created, token), status_code=201)(scope, receive, send)
                return
            if path == "/session":
                await JSONResponse(session_info(tenant))(scope, receive, send)
                return
            if path in {"/models", "/recipes"}:
                from opendpd.core.registry import list_models
                from opendpd.services.recipes import list_recipes
                models = [m for m in list_models() if m.status == "supported" and m.training_method == "gradient"]
                keys = {m.key for m in models}
                payload = [m.to_dict() for m in models] if path == "/models" else [r.to_dict() for r in list_recipes() if r.model.key in keys]
                await JSONResponse(payload)(scope, receive, send)
                return
            if method in {"POST", "PUT"}:
                check_body(path, body)
            if tenant.closing or self.manager.now() >= tenant.expires_at:
                reject(401, "session_expired", "the temporary workspace has expired")
            # Each nested app owns its own Workspace, RunStore, Supervisor and local
            # session. No process-global app.state is ever swapped per request.
            internal_headers = [(b"host", b"127.0.0.1"), (b"content-type", b"application/json"),
                                (b"cookie", f"{SESSION_COOKIE}={tenant.local_session.session_id}".encode()),
                                (CSRF_HEADER.encode(), tenant.local_session.csrf_token.encode())]
            child_scope = {**scope, "scheme": "http", "headers": internal_headers, "state": {}}
            sent = False

            async def child_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": data, "more_body": False}
                return await receive()

            await tenant.app(child_scope, child_receive, send)
        finally:
            self.manager.inflight -= 1
            if tenant:
                tenant.inflight -= 1


def create_web_app(config: WebConfig, *, now=None):
    manager = TenantManager(config, **({"now": now} if now is not None else {}))

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        maintenance = asyncio.create_task(manager.maintenance())
        try:
            yield
        finally:
            maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await maintenance
            await manager.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = manager
    app.add_middleware(PublicBoundary, manager=manager)
    return app
