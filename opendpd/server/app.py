"""Application factory and development server entry point."""

from __future__ import annotations

import html
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from opendpd import __version__
from opendpd.runtime.db import RunStore
from opendpd.runtime.supervisor import Supervisor
from opendpd.server.security import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    DatasetImportBoundary,
    LocalBoundaryMiddleware,
    SessionStore,
)
from opendpd.services.config import ConfigError
from opendpd.services.workspace import Workspace, WorkspaceError

log = logging.getLogger("opendpd.server")
STATIC_DIR = Path(__file__).resolve().parents[1] / "studio" / "static"
API_PREFIX = "/api/v1"


def error_payload(code: str, message: str, details=None, hint: Optional[str] = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or [], "hint": hint}}


def static_status(static_dir: Path = STATIC_DIR) -> dict:
    """Presence and version of the built frontend (never a white page)."""
    index = static_dir / "index.html"
    if not index.is_file():
        return {"present": False, "version": None, "problem": "frontend assets are not built/installed"}
    info_path = static_dir / "build-info.json"
    version = None
    if info_path.is_file():
        try:
            version = json.loads(info_path.read_text()).get("opendpd_version")
        except (OSError, ValueError):
            version = None
    problem = None
    if version is not None and version != __version__:
        problem = f"frontend build {version} does not match opendpd {__version__}"
    return {"present": True, "version": version, "problem": problem}


def create_app(workspace_root: Path, *, bootstrap_token: Optional[str] = None, static_dir: Path = STATIC_DIR,
               supervisor_kwargs: Optional[dict] = None, shutdown_timeout: float = 10.0,
               allow_custom_datasets: bool = False) -> FastAPI:
    sessions = SessionStore(bootstrap_token)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ws = Workspace.open_or_create(Path(workspace_root))
        store = RunStore(ws.root / "metadata.sqlite")
        supervisor = Supervisor(ws, store, **(supervisor_kwargs or {}))
        supervisor.start()
        app.state.ws, app.state.store, app.state.supervisor = ws, store, supervisor
        try:
            yield
        finally:
            supervisor.stop(timeout=shutdown_timeout)
            store.close()

    app = FastAPI(title="OpenDPD Studio API", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=f"{API_PREFIX}/openapi.json")
    app.state.sessions = sessions
    app.state.static_dir = Path(static_dir)
    app.state.allow_custom_datasets = allow_custom_datasets
    app.add_middleware(DatasetImportBoundary, enabled=allow_custom_datasets)
    app.add_middleware(LocalBoundaryMiddleware)

    # -- uniform error shape --------------------------------------------------
    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            payload = detail
        else:
            payload = error_payload("http_error", str(detail))
        return JSONResponse(payload, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
                   for e in exc.errors()]
        return JSONResponse(error_payload("invalid_request", "request validation failed", details), status_code=422)

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: ConfigError):
        details = [{"field": i.field, "message": i.message, "hint": i.hint} for i in exc.issues]
        return JSONResponse(error_payload("invalid_config", "the experiment configuration is not valid", details,
                                          hint="fix the listed fields; nothing was started"), status_code=422)

    @app.exception_handler(WorkspaceError)
    async def _workspace_error(request: Request, exc: WorkspaceError):
        return JSONResponse(error_payload("workspace_error", str(exc)), status_code=409)

    # -- liveness / readiness --------------------------------------------------
    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request):
        problems = []
        ws = getattr(request.app.state, "ws", None)
        if ws is None:
            problems.append("workspace not open")
        else:
            problems.extend(ws.preflight())
        sup = getattr(request.app.state, "supervisor", None)
        if sup is None or not sup.alive:
            problems.append("supervisor not running")
        static = static_status(request.app.state.static_dir)
        if static["problem"]:
            problems.append(static["problem"])
        payload = {"ready": not problems, "problems": problems, "version": __version__, "frontend": static}
        return JSONResponse(payload, status_code=200 if not problems else 503)

    @app.get("/bootstrap", include_in_schema=False)
    async def bootstrap(request: Request, token: str = ""):
        session = request.app.state.sessions.exchange(token)
        if session is None:
            return HTMLResponse(_diagnostic_page("This link is not valid for the running OpenDPD Studio server.",
                                                 "Start it again with `opendpd gui` and use the URL it prints."),
                                status_code=401)
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(SESSION_COOKIE, session.session_id, max_age=SESSION_MAX_AGE, httponly=True,
                            samesite="strict", path="/")
        return response

    from opendpd.server.routes import router
    app.include_router(router, prefix=API_PREFIX)

    # -- static frontend with SPA fallback; API paths never fall back --------------
    assets = Path(static_dir) / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(request: Request, path: str):
        if path.startswith("api/") or path == "api":
            return JSONResponse(error_payload("not_found", f"unknown API route /{path}"), status_code=404)
        status = static_status(request.app.state.static_dir)
        if not status["present"] or status["problem"]:
            return HTMLResponse(_diagnostic_page("OpenDPD Studio frontend is not available.",
                                                 status["problem"] or "frontend assets are not built/installed"),
                                status_code=503)
        index = request.app.state.static_dir / "index.html"
        return HTMLResponse(index.read_text(encoding="utf-8"))

    return app


def _diagnostic_page(title: str, detail: str) -> str:
    title, detail = html.escape(title), html.escape(detail)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>OpenDPD Studio</title></head>"
            f"<body style='font-family:system-ui;margin:3rem;max-width:40rem'><h1>{title}</h1>"
            f"<p>{detail}</p><p>The API is running (opendpd {__version__}); see <code>/readyz</code>.</p>"
            f"</body></html>")
