"""Public ASGI boundary with isolated local-app instances and a private GPU bridge."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse, Response

from opendpd import __version__
from opendpd.server.security import CSRF_HEADER, SESSION_COOKIE
from opendpd.web.policy import WebConfig, allowed, check_body, check_query, reject
from opendpd.web.runtime import TenantManager, directory_bytes
from opendpd.web.gpu_archive import MAX_BYTES

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
        if request.url.path.startswith("/_gpu/"):
            await self.gpu_request(request, scope, receive, send)
            return
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
        self.manager.rate_limit(ip_key + (":preflight" if request.method == "OPTIONS" else ""))
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
        if tenant:
            self.manager.rate_limit("session:" + tenant.identifier, self.config.requests_per_session_minute)
        if self.manager.inflight >= self.config.max_requests or (tenant and tenant.inflight >= 3):
            reject(429, "service_busy", "too many simultaneous requests; try again later")
        # Reserve before receiving the body, including chunked and slow requests.
        self.manager.inflight += 1
        if tenant:
            tenant.inflight += 1
        try:
            if path == '/datasets/upload' and method == 'POST':
                await self.upload_csv(request, tenant, scope, receive, send)
                return
            data = b""
            length = headers.get("content-length")
            if length is not None and (not length.isdecimal() or int(length) > self.config.max_body):
                reject(413, "payload_too_large", "request body exceeds the public API limit")
            if method in {"POST", "PUT"} and headers.get("content-type", "").split(";")[0] != "application/json":
                reject(415, "json_required", "this endpoint accepts JSON requests only")

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
            if path in {'/datasets/csv', '/datasets/csv/preview'}:
                from opendpd.services.csv_upload import CsvUploadRejected, validated_source
                self.manager.rate_limit('csv-scan:' + tenant.ip_key, 20)
                try:
                    _, validation = await asyncio.to_thread(validated_source, tenant.app.state.ws, body.get('source'))
                except (CsvUploadRejected, ValueError, KeyError, OSError) as exc:
                    reject(422, 'csv_rejected', str(exc) if isinstance(exc, CsvUploadRejected) else 'The validated CSV is unavailable. Upload it again.')
                if path == '/datasets/csv':
                    used = await asyncio.to_thread(directory_bytes, tenant.root)
                    # Bound the source copy, arrays, split CSVs and preprocessing
                    # metadata before materialising a newly uploaded dataset.
                    if used + validation['n_samples'] * 160 > self.config.max_workspace_bytes:
                        reject(413, 'workspace_limit', 'This CSV would exceed temporary workspace storage. Use fewer samples.')
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

    async def upload_csv(self, request, tenant, scope, receive, send):
        from opendpd.services.csv_upload import MAX_UPLOAD_BYTES, CsvUploadRejected, admit_upload, check_filename, quarantine_path
        if tenant.uploading:
            reject(429, 'upload_busy', 'Wait for the current CSV upload to finish.')
        self.manager.rate_limit('csv-upload:' + tenant.ip_key, 6)
        if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'text/csv':
            reject(415, 'csv_required', 'Only CSV file uploads are accepted.')
        try:
            check_filename(request.query_params.get('filename', ''))
        except CsvUploadRejected as exc:
            reject(422, 'csv_rejected', str(exc))
        length = request.headers.get('content-length')
        if length is not None and (not length.isdecimal() or int(length) > MAX_UPLOAD_BYTES):
            reject(413, 'payload_too_large', 'CSV files must be at most 25 MiB.')
        tenant.uploading = True
        path = None
        try:
            used = await asyncio.to_thread(directory_bytes, tenant.root)
            if used + MAX_UPLOAD_BYTES > self.config.max_workspace_bytes:
                reject(413, 'workspace_limit', 'Temporary workspace storage is full.')
            ws = tenant.app.state.ws
            if len(list((ws.imports_dir / 'uploads').glob('*.csv'))) >= 4:
                reject(429, 'upload_limit', 'This workspace already has four uploaded CSV files.')
            path = quarantine_path(ws)

            async def receive_csv():
                size = 0
                with path.open('xb') as output:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            reject(413, 'payload_too_large', 'CSV size limit exceeded. Upload deleted.')
                        output.write(chunk)
                if length is not None and size != int(length):
                    reject(400, 'incomplete_upload', 'CSV upload was incomplete. Upload deleted.')

            await asyncio.wait_for(receive_csv(), timeout=45)
            if tenant.closing or self.manager.now() >= tenant.expires_at:
                reject(401, 'session_expired', 'The temporary workspace expired. Upload deleted.')
            # Shield validation from request cancellation: wait for the bounded
            # scan before cleanup so no background thread can publish afterward.
            scan = asyncio.create_task(asyncio.to_thread(admit_upload, ws, path))
            try:
                result = await asyncio.shield(scan)
            except asyncio.CancelledError:
                await scan
                raise
            await JSONResponse(result, status_code=201)(scope, receive, send)
        except CsvUploadRejected as exc:
            reject(422, 'csv_rejected', str(exc))
        except asyncio.TimeoutError:
            reject(408, 'request_timeout', 'CSV upload took too long. Upload deleted.')
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            tenant.uploading = False

    async def gpu_request(self, request, scope, receive, send):
        # This path is outside the Cloudflare Tunnel ingress. It also requires a
        # separate secret and the host-local Host header; browser requests fail closed.
        headers = request.headers
        token = self.config.gpu_token
        if (not token or request.url.hostname != "127.0.0.1" or len(headers.getlist("host")) != 1 or headers.get("origin")
                or headers.get("cf-connecting-ip") or headers.get("x-forwarded-proto")
                or len(headers.getlist("x-opendpd-gpu")) != 1
                or not secrets.compare_digest(headers.get("x-opendpd-gpu", ""), token)):
            reject(403, "private_endpoint", "private compute endpoint")
        path = request.url.path.split("/")
        broker = self.manager.gpu
        if request.method == "POST":
            limit = MAX_BYTES if path[-1] == "result" else 24 * 1024 * 1024 if path[-1] == 'checkpoint' else 4 * 1024 * 1024
            async def read():
                data = bytearray()
                async for chunk in request.stream():
                    data.extend(chunk)
                    if len(data) > limit:
                        reject(413, "transfer_limit", "GPU transfer exceeds limit")
                return bytes(data)
            raw = await asyncio.wait_for(read(), timeout=20)
        else:
            raw = b""
        try:
            if request.url.path == "/_gpu/poll" and request.method == "POST":
                body = json.loads(raw)
                result = {"job": broker.poll(body["name"])}
            else:
                if len(path) != 5 or path[2] != "jobs":
                    raise ValueError("unknown GPU operation")
                with broker.lock:
                    job = broker.get(path[3], headers.get("x-opendpd-lease", ""))
                if path[4] == "input" and request.method == "GET":
                    data = await asyncio.to_thread(broker.input, job)
                    await Response(data, media_type="application/zip")(scope, receive, send)
                    return
                if path[4] == "update" and request.method == "POST":
                    result = broker.update(job, json.loads(raw))
                elif path[4] == 'checkpoint' and request.method == 'POST':
                    result = await asyncio.to_thread(broker.checkpoint, job, json.loads(raw))
                elif path[4] == "result" and request.method == "POST":
                    code = int(headers.get("x-opendpd-exit", "1"))
                    if code not in {0, 1, 3}:
                        raise ValueError("invalid exit code")
                    await asyncio.to_thread(broker.result, job, raw, code)
                    result = {"ok": True}
                else:
                    raise ValueError("unknown GPU operation")
            await JSONResponse(result)(scope, receive, send)
        except (ValueError, KeyError, TypeError):
            reject(409, "invalid_gpu_transfer", "invalid or expired GPU transfer")


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
