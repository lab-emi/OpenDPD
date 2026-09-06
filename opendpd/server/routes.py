"""API routes under /api/v1. All state changes go through the supervisor."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from opendpd import __version__
from opendpd.core.registry import list_models
from opendpd.schemas import (
    ArtifactManifest,
    DatasetManifest,
    EvaluationResult,
    ExperimentConfig,
    ModelSpec,
    ResolvedExperimentConfig,
    RunEvent,
    RunRecord,
    RunStatus,
    TERMINAL_STATUSES,
    TaskType,
    TrainingConfig,
    heartbeat_is_stale,
    utcnow,
)
from opendpd.server.security import CSRF_HEADER, SESSION_COOKIE, SESSION_MAX_AGE, Session
from opendpd.services import experiments
from opendpd.services.config import ConfigError, ConfigIssue, validate as validate_config
from opendpd.services.workspace import WorkspaceError
from opendpd.services.recipes import list_recipes

router = APIRouter()
HEARTBEAT_TIMEOUT = timedelta(seconds=60)


def _error(status: int, code: str, message: str, hint: Optional[str] = None, details=None) -> HTTPException:
    return HTTPException(status_code=status,
                         detail={"error": {"code": code, "message": message, "details": details or [], "hint": hint}})


# --- session ---------------------------------------------------------------------

def require_session(request: Request) -> Session:
    session = request.app.state.sessions.get(request.cookies.get(SESSION_COOKIE))
    if session is None:
        raise _error(401, "unauthorized", "no valid local session",
                     hint="open the URL printed by `opendpd gui` (it carries the bootstrap token)")
    return session


def require_csrf(request: Request, session: Session = Depends(require_session)) -> Session:
    header = request.headers.get(CSRF_HEADER)
    if not header or header != session.csrf_token:
        raise _error(403, "csrf_required", f"state-changing requests need the {CSRF_HEADER} header",
                     hint="read the token from GET /api/v1/session")
    return session


class BootstrapRequest(BaseModel):
    token: str


class SessionInfo(BaseModel):
    authenticated: bool
    csrf_token: Optional[str] = None
    version: str = __version__


@router.post("/session/bootstrap", response_model=SessionInfo, tags=["session"])
def session_bootstrap(body: BootstrapRequest, request: Request, response: Response):
    session = request.app.state.sessions.exchange(body.token)
    if session is None:
        raise _error(401, "bad_bootstrap_token", "the bootstrap token is not valid for this server")
    response.set_cookie(SESSION_COOKIE, session.session_id, max_age=SESSION_MAX_AGE, httponly=True,
                        samesite="strict", path="/")
    return SessionInfo(authenticated=True, csrf_token=session.csrf_token)


@router.get("/session", response_model=SessionInfo, tags=["session"])
def session_info(request: Request):
    session = request.app.state.sessions.get(request.cookies.get(SESSION_COOKIE))
    if session is None:
        return SessionInfo(authenticated=False)
    return SessionInfo(authenticated=True, csrf_token=session.csrf_token)


# --- system -----------------------------------------------------------------------

class DeviceInfo(BaseModel):
    device: str
    detected: bool
    name: Optional[str] = None
    count: int = 0
    tested_models: List[str] = Field(default_factory=list)   # models with recorded evidence on this device


class Capabilities(BaseModel):
    version: str
    devices: List[DeviceInfo]
    workspace: str
    note: str


_device_cache: Dict[str, Any] = {}


def _detect_devices() -> Dict[str, Any]:
    if _device_cache:
        return _device_cache
    info = {"cuda": {"detected": False, "count": 0, "name": None}, "mps": {"detected": False}}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = {"detected": True, "count": torch.cuda.device_count(),
                            "name": torch.cuda.get_device_name(0)}
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = {"detected": bool(mps and mps.is_available())}
    except Exception as err:  # noqa: BLE001 - torch missing or broken driver
        info["error"] = str(err)
    _device_cache.update(info)
    return _device_cache


@router.get("/system/capabilities", response_model=Capabilities, tags=["system"],
            dependencies=[Depends(require_session)])
def capabilities(request: Request):
    detected = _detect_devices()
    models = list_models()
    devices = [DeviceInfo(device="cpu", detected=True, count=1,
                          tested_models=[m.key for m in models if "cpu" in m.devices_tested])]
    for dev in ("cuda", "mps"):
        d = detected.get(dev, {})
        devices.append(DeviceInfo(device=dev, detected=bool(d.get("detected")), name=d.get("name"),
                                  count=int(d.get("count", 1 if d.get("detected") else 0)),
                                  tested_models=[m.key for m in models if dev in m.devices_tested]))
    return Capabilities(version=__version__, devices=devices, workspace=str(request.app.state.ws.root),
                        note="'detected' means the driver reports the device; 'tested_models' lists models "
                             "with recorded evidence on it. One does not imply the other.")


class ParamSpecInfo(BaseModel):
    name: str
    type: str
    default: Any
    description: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[List[Any]] = None
    legacy_arg: Dict[str, str] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Registry descriptor as seen by clients (opendpd.core.registry.ModelDescriptor)."""
    key: str
    display_name: str
    family: str
    legacy_backbone: str
    training_method: str
    roles: List[str]
    params: List[ParamSpecInfo]
    status: str
    devices_tested: List[str]
    lookahead_samples: Optional[int] = None
    lookahead_note: str
    execution_semantics: str
    export_formats: List[str]
    constraints: Optional[str] = None
    reference: Optional[str] = None
    evidence: Optional[str] = None


class RecipeInfo(BaseModel):
    recipe_id: str
    title: str
    purpose: str
    task: TaskType
    model: ModelSpec
    training: TrainingConfig
    description: str
    limits: str
    expected_duration: str


@router.get("/models", response_model=List[ModelInfo], tags=["system"], dependencies=[Depends(require_session)])
def models():
    return [m.to_dict() for m in list_models()]


@router.get("/recipes", response_model=List[RecipeInfo], tags=["system"], dependencies=[Depends(require_session)])
def recipes():
    return [r.to_dict() for r in list_recipes()]


# --- datasets ------------------------------------------------------------------------

class ImportBuiltinRequest(BaseModel):
    name: str
    dataset_id: Optional[str] = None


@router.get("/datasets", response_model=List[DatasetManifest], tags=["datasets"],
            dependencies=[Depends(require_session)])
def datasets_list(request: Request):
    return request.app.state.ws.list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetManifest, tags=["datasets"],
            dependencies=[Depends(require_session)])
def dataset_get(dataset_id: str, request: Request):
    return request.app.state.ws.get_dataset(dataset_id)


@router.post("/datasets/import-builtin", response_model=DatasetManifest, status_code=201, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_import_builtin(body: ImportBuiltinRequest, request: Request):
    return request.app.state.ws.register_builtin_dataset(body.name, dataset_id=body.dataset_id)


# --- experiments and runs ------------------------------------------------------------

class ValidateRequest(BaseModel):
    config: Dict[str, Any]


@router.post("/experiments/validate", tags=["experiments"], dependencies=[Depends(require_session)])
def experiments_validate(body: ValidateRequest, request: Request) -> Dict[str, Any]:
    """Resolve and bind without starting anything; config problems are data, not 4xx."""
    report = validate_config(body.config)
    if report.ok:
        ws = request.app.state.ws
        try:
            ws.get_dataset(report.resolved.dataset.id)
            bound = experiments.bind_references(ws, ExperimentConfig.model_validate(body.config))
            report = validate_config(bound.model_dump(mode="json"))
        except ConfigError as err:
            report.errors.extend(err.issues)
            report.resolved = None
        except WorkspaceError as err:
            report.errors.append(ConfigIssue("dataset.id", str(err), hint="import the dataset first"))
            report.resolved = None
    return report.to_dict()


class SubmitRunRequest(BaseModel):
    config: ExperimentConfig
    name: Optional[str] = None
    idempotency_key: Optional[str] = Field(default=None, max_length=256)


class RunView(RunRecord):
    heartbeat_stale: bool = False


def _view(record: RunRecord) -> RunView:
    return RunView(**record.model_dump(), heartbeat_stale=heartbeat_is_stale(record, utcnow(), HEARTBEAT_TIMEOUT))


@router.get("/runs", response_model=List[RunView], tags=["runs"], dependencies=[Depends(require_session)])
def runs_list(request: Request, status: Optional[RunStatus] = None, limit: int = Query(50, ge=1, le=500),
              offset: int = Query(0, ge=0)):
    return [_view(r) for r in request.app.state.store.list_runs(status=status, limit=limit, offset=offset)]


@router.post("/runs", response_model=RunView, status_code=201, tags=["runs"], dependencies=[Depends(require_csrf)])
def runs_submit(body: SubmitRunRequest, request: Request, response: Response):
    sup = request.app.state.supervisor
    if body.idempotency_key:
        existing = request.app.state.store.find_idempotent(body.idempotency_key)
        if existing is not None:
            response.status_code = 200
            return _view(existing)
    try:
        record = sup.submit(body.config, name=body.name, idempotency_key=body.idempotency_key)
    except RuntimeError as err:
        raise _error(503, "shutting_down", str(err))
    return _view(record)


def _get_run(request: Request, run_id: str) -> RunRecord:
    record = request.app.state.store.get_run(run_id)
    if record is None:
        raise _error(404, "run_not_found", f"run '{run_id}' does not exist")
    return record


@router.get("/runs/{run_id}", response_model=RunView, tags=["runs"], dependencies=[Depends(require_session)])
def runs_get(run_id: str, request: Request):
    return _view(_get_run(request, run_id))


@router.post("/runs/{run_id}/cancel", response_model=RunView, tags=["runs"], dependencies=[Depends(require_csrf)])
def runs_cancel(run_id: str, request: Request):
    _get_run(request, run_id)
    return _view(request.app.state.supervisor.cancel(run_id))


@router.post("/runs/{run_id}/retry", response_model=RunView, status_code=201, tags=["runs"],
             dependencies=[Depends(require_csrf)])
def runs_retry(run_id: str, request: Request):
    record = _get_run(request, run_id)
    if record.status not in TERMINAL_STATUSES:
        raise _error(409, "run_not_finished", "only finished runs can be retried")
    return _view(request.app.state.supervisor.retry(run_id))


@router.get("/runs/{run_id}/config", response_model=ResolvedExperimentConfig, tags=["runs"],
            dependencies=[Depends(require_session)])
def runs_config(run_id: str, request: Request):
    _get_run(request, run_id)
    return experiments.load_resolved(request.app.state.ws, run_id)


@router.get("/runs/{run_id}/artifacts", response_model=ArtifactManifest, tags=["runs"],
            dependencies=[Depends(require_session)])
def runs_artifacts(run_id: str, request: Request):
    _get_run(request, run_id)
    manifest = experiments.load_artifacts(request.app.state.ws, run_id)
    return manifest or ArtifactManifest(run_id=run_id)


@router.get("/results/{run_id}", response_model=EvaluationResult, tags=["results"],
            dependencies=[Depends(require_session)])
def results_get(run_id: str, request: Request):
    record = _get_run(request, run_id)
    result = experiments.load_result(request.app.state.ws, run_id)
    if result is None:
        raise _error(404, "result_not_available", f"run '{run_id}' has no formal result (status {record.status.value})",
                     hint="results exist only for succeeded train_pa / train_dpd runs")
    return result


# --- events: replayable list, SSE stream --------------------------------------------------

class EventPage(BaseModel):
    events: List[RunEvent]
    last_seq: int
    terminal: bool


def _check_cursor(request: Request, run_id: str, after: int) -> RunRecord:
    record = _get_run(request, run_id)
    if after > record.last_event_seq:
        raise _error(409, "cursor_out_of_range",
                     f"cursor {after} is beyond the last event ({record.last_event_seq}); resynchronise from a snapshot",
                     hint="GET /runs/{id} then GET /runs/{id}/events/list?after=0")
    return record


@router.get("/runs/{run_id}/events/list", response_model=EventPage, tags=["events"],
            dependencies=[Depends(require_session)])
def events_list(run_id: str, request: Request, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000)):
    record = _check_cursor(request, run_id, after)
    events = request.app.state.store.events_after(run_id, after, limit)
    return EventPage(events=events, last_seq=events[-1].seq if events else after,
                     terminal=record.status in TERMINAL_STATUSES)


@router.get("/runs/{run_id}/events", tags=["events"], dependencies=[Depends(require_session)])
async def events_stream(run_id: str, request: Request, after: int = Query(0, ge=0)):
    """Server-Sent Events: ``id`` is the event seq; ``event: end`` closes a finished run."""
    _check_cursor(request, run_id, after)
    store = request.app.state.store
    last_event_id = request.headers.get("last-event-id")
    cursor = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

    async def generate():
        nonlocal cursor
        idle = 0.0
        while True:
            if await request.is_disconnected():
                return
            events = store.events_after(run_id, cursor, 500)
            for e in events:
                cursor = e.seq
                yield f"id: {e.seq}\nevent: {e.type.value}\ndata: {e.model_dump_json()}\n\n"
            record = store.get_run(run_id)
            if record is not None and record.status in TERMINAL_STATUSES and not store.events_after(run_id, cursor, 1):
                yield f"event: end\ndata: {json.dumps({'status': record.status.value, 'last_seq': cursor})}\n\n"
                return
            if events:
                idle = 0.0
            else:
                idle += 0.5
                if idle >= 15.0:
                    yield ": keepalive\n\n"
                    idle = 0.0
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- logs and artifacts -----------------------------------------------------------------------

class LogPage(BaseModel):
    lines: List[str]
    next_offset: int
    eof: bool
    size: int


@router.get("/runs/{run_id}/logs", response_model=LogPage, tags=["runs"], dependencies=[Depends(require_session)])
def runs_logs(run_id: str, request: Request, offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=2000)):
    """Page through logs/worker.log by byte offset (safe for very large logs).

    A trailing line without newline is returned only once the run has finished
    (while running it may still be written).
    """
    record = _get_run(request, run_id)
    finished = record.status in TERMINAL_STATUSES
    path = request.app.state.ws.run_dir(run_id) / "logs" / "worker.log"
    if not path.exists():
        return LogPage(lines=[], next_offset=0, eof=True, size=0)
    size = path.stat().st_size
    pos = min(offset, size)
    lines: List[str] = []
    with open(path, "rb") as f:
        f.seek(pos)
        while len(lines) < limit:
            raw = f.readline()
            if not raw or (not raw.endswith(b"\n") and not finished):
                break
            pos += len(raw)
            lines.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
    return LogPage(lines=lines, next_offset=pos, eof=pos >= size, size=size)


@router.get("/artifacts/{run_id}/{artifact_id}", tags=["artifacts"], dependencies=[Depends(require_session)])
def artifact_download(run_id: str, artifact_id: str, request: Request):
    """Download by *registered id* only; paths are never accepted from clients."""
    _get_run(request, run_id)
    manifest = experiments.load_artifacts(request.app.state.ws, run_id)
    artifact = next((a for a in (manifest.artifacts if manifest else []) if a.artifact_id == artifact_id), None)
    if artifact is None:
        raise _error(404, "artifact_not_found", f"run '{run_id}' has no artifact '{artifact_id}'")
    run_dir = request.app.state.ws.run_dir(run_id).resolve()
    path = (run_dir / artifact.file.path).resolve()
    if run_dir not in path.parents or not path.is_file():
        raise _error(404, "artifact_missing", "the registered file is outside the run directory or missing")
    return FileResponse(str(path), filename=Path(artifact.file.path).name, media_type="application/octet-stream")
