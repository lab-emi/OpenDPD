"""API routes under /api/v1. All state changes go through the supervisor."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from opendpd import __version__
from opendpd.core.metrics import get_profile, list_profiles
from opendpd.core.registry import list_models
from opendpd.core.splits import DEFAULT_GUARD_SAMPLES
from opendpd.schemas import (
    AdaptationReport,
    DeploymentManifest,
    AdaptationReportSummary,
    ArtifactManifest,
    ComparisonReport,
    DatasetManifest,
    DatasetOrigin,
    DatasetVersion,
    DiagnosticReport,
    EvaluationResult,
    ExperimentConfig,
    HistoryPoint,
    ImportReport,
    ModelSpec,
    PackageManifest,
    PreprocessingParams,
    ResolvedExperimentConfig,
    RunEvent,
    RunLineage,
    RunRecord,
    RunStatus,
    TERMINAL_STATUSES,
    SignalSpec,
    TaskType,
    TrainingConfig,
    heartbeat_is_stale,
    utcnow,
    MetricProfile,
    WorkspaceSettings,
    UILanguage,
)
from opendpd.server.security import CSRF_HEADER, SESSION_COOKIE, SESSION_MAX_AGE, UPLOAD_MAX_BODY, Session
from opendpd.services import adaptation as adaptation_service
from opendpd.services import deploy as deploy_service
from opendpd.services import capabilities as capabilities_service
from opendpd.services import datasets as datasets_service
from opendpd.services.dataset_analysis import analyze_dataset
from opendpd.schemas.analysis import DatasetAnalysis
from opendpd.schemas.importing import BuiltinDatasetInfo, CsvInspection, CsvOptions, DatasetImportDefaults
from opendpd.schemas.common import Slug, Sha256
from opendpd.services import experiments
from opendpd.services import packages
from opendpd.services.evaluation import available_profiles, compare_results, comparison_csv
from opendpd.services.packages import PackageError
from opendpd.services.reports import report_html, report_markdown
from opendpd.services.workspace import WorkspaceError, list_builtin_datasets
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

@router.get("/system/about", response_model=Dict[str, Any], tags=["system"],
            dependencies=[Depends(require_session)])
def system_about():
    from opendpd.services.about import project_info
    return project_info()

# --- workbench settings -----------------------------------------------------------

@router.get("/settings", response_model=WorkspaceSettings, tags=["settings"], dependencies=[Depends(require_session)])
def settings_get(request: Request):
    return _ws(request).settings()


@router.put("/settings", response_model=WorkspaceSettings, tags=["settings"], dependencies=[Depends(require_csrf)])
def settings_put(body: WorkspaceSettings, request: Request):
    return _ws(request).save_settings(body)


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
    custom_dataset_imports: bool = False


@router.get("/system/capabilities", response_model=Capabilities, tags=["system"],
            dependencies=[Depends(require_session)])
def capabilities(request: Request):
    detected = getattr(request.app.state, "device_detector", capabilities_service.detect_devices)()
    models = list_models()
    devices = [DeviceInfo(device="cpu", detected=True, count=1,
                          tested_models=[m.key for m in models if "cpu" in m.devices_tested])]
    for dev in ("cuda", "mps"):
        d = detected.get(dev, {})
        devices.append(DeviceInfo(device=dev, detected=bool(d.get("detected")), name=d.get("name"),
                                  count=int(d.get("count", 1 if d.get("detected") else 0)),
                                  tested_models=[m.key for m in models if dev in m.devices_tested]))
    return Capabilities(version=__version__, devices=devices,
                        workspace=getattr(request.app.state, "workspace_label", None) or str(request.app.state.ws.root),
                        custom_dataset_imports=request.app.state.allow_custom_datasets,
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
    weights_from: Optional[str] = None
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


class ImportRootInfo(BaseModel):
    root_id: str
    path: str
    exists: bool


class FileEntryInfo(BaseModel):
    path: str
    kind: str
    size_bytes: int = 0


class SourceRef(BaseModel):
    """A file inside an authorised import root; clients never send absolute paths."""
    root_id: str
    path: str = Field(max_length=1024)


class SourceInfoOut(BaseModel):
    kind: str
    path: str
    columns: List[str]
    suggested_mapping: Dict[str, str]
    n_rows: Optional[int] = None
    preview: List[Dict[str, float]]
    arrays: Dict[str, Dict[str, Any]]
    problems: List[str]
    legacy_files: List[str]


class ImportRequest(BaseModel):
    source: SourceRef
    dataset_id: Optional[str] = Field(default=None, max_length=64)
    display_name: Optional[str] = Field(default=None, max_length=200)
    mapping: Dict[str, str] = Field(default_factory=dict)
    signal: SignalSpec = SignalSpec()
    origin: DatasetOrigin = DatasetOrigin.unknown
    guard_samples: int = Field(default=DEFAULT_GUARD_SAMPLES, ge=0, le=100_000)
    notes: Optional[str] = Field(default=None, max_length=2000)


class CsvPreviewRequest(BaseModel):
    source: SourceRef
    options: CsvOptions = Field(default_factory=CsvOptions)
    split: DatasetImportDefaults = Field(default_factory=DatasetImportDefaults)
    signal: SignalSpec = Field(default_factory=SignalSpec)


class CsvCreateRequest(CsvPreviewRequest):
    dataset_id: Slug
    display_name: str = Field(min_length=1, max_length=200)
    signal: SignalSpec = Field(default_factory=SignalSpec)
    origin: DatasetOrigin = DatasetOrigin.unknown
    expected_sha256: Sha256


class ManifestUpdate(BaseModel):
    signal: Optional[SignalSpec] = None
    display_name: Optional[str] = Field(default=None, max_length=200)
    origin: Optional[DatasetOrigin] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


class PreprocessRequest(BaseModel):
    params: PreprocessingParams
    base_version: str = "raw-v1"
    version: Optional[str] = Field(default=None, max_length=64)   # required to create, ignored for preview


class PreprocessPreview(BaseModel):
    n_samples_before: int
    n_samples_after: int
    record: Dict[str, Any]
    report_after: DiagnosticReport


class UploadResult(BaseModel):
    root_id: str
    path: str
    size_bytes: int


def _ws(request: Request):
    return request.app.state.ws


@router.get("/datasets", response_model=List[DatasetManifest], tags=["datasets"],
            dependencies=[Depends(require_session)])
def datasets_list(request: Request):
    return _ws(request).list_datasets()


@router.get("/datasets/import-roots", response_model=List[ImportRootInfo], tags=["datasets"],
            dependencies=[Depends(require_session)])
def import_roots(request: Request):
    return [ImportRootInfo(root_id=k, path=str(v), exists=v.exists()) for k, v in _ws(request).import_roots().items()]


@router.get("/datasets/builtin", response_model=List[BuiltinDatasetInfo], tags=["datasets"],
            dependencies=[Depends(require_session)])
def datasets_builtin():
    return list_builtin_datasets()


@router.get("/datasets/import-defaults", response_model=DatasetImportDefaults, tags=["datasets"],
            dependencies=[Depends(require_session)])
def dataset_import_defaults():
    return DatasetImportDefaults()


@router.get("/datasets/import-roots/{root_id}/files", response_model=List[FileEntryInfo], tags=["datasets"],
            dependencies=[Depends(require_session)])
def import_root_files(root_id: str, request: Request, path: str = Query("", max_length=1024)):
    return [FileEntryInfo(**vars(e)) for e in datasets_service.list_files(_ws(request), root_id, path)]


@router.post("/datasets/inspect", response_model=SourceInfoOut, tags=["datasets"], dependencies=[Depends(require_session)])
def dataset_inspect(body: SourceRef, request: Request):
    target = datasets_service.resolve_in_root(_ws(request), body.root_id, body.path)
    if not target.exists():
        raise _error(404, "source_not_found", f"{body.path} does not exist in root '{body.root_id}'")
    info = datasets_service.inspect_source(target)
    return SourceInfoOut(kind=info.kind.value, path=body.path, columns=info.columns, suggested_mapping=info.suggested_mapping,
                         n_rows=info.n_rows, preview=info.preview, arrays=info.arrays, problems=info.problems,
                         legacy_files=info.legacy_files)


@router.post("/datasets/import", response_model=DatasetManifest, status_code=201, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_import(body: ImportRequest, request: Request):
    ws = _ws(request)
    target = datasets_service.resolve_in_root(ws, body.source.root_id, body.source.path)
    return datasets_service.import_dataset(ws, target, dataset_id=body.dataset_id, display_name=body.display_name,
                                           mapping=body.mapping, signal=body.signal, origin=body.origin,
                                           guard_samples=body.guard_samples, notes=body.notes)


@router.post("/datasets/upload", response_model=UploadResult, status_code=201, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
async def dataset_upload(request: Request, file: UploadFile):
    """Browser upload, streamed into <workspace>/imports/uploads (then imported like any root file)."""
    ws = _ws(request)

    def chunks():
        while True:
            chunk = file.file.read(1 << 20)
            if not chunk:
                return
            yield chunk

    try:
        path = datasets_service.receive_upload(ws, file.filename or "upload.csv", chunks(), UPLOAD_MAX_BODY)
    except datasets_service.UploadTooLarge as err:
        raise _error(413, "payload_too_large", str(err))
    return UploadResult(root_id="imports", path=path.relative_to(ws.imports_dir).as_posix(), size_bytes=path.stat().st_size)


@router.post("/datasets/csv/preview", response_model=CsvInspection, tags=["datasets"],
             dependencies=[Depends(require_session)])
def dataset_csv_preview(body: CsvPreviewRequest, request: Request):
    from opendpd.services.csv_import import inspect_csv
    target = datasets_service.resolve_in_root(_ws(request), body.source.root_id, body.source.path)
    return inspect_csv(target, body.options, body.split)[0]


@router.post("/datasets/csv", response_model=DatasetManifest, status_code=201, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_csv_create(body: CsvCreateRequest, request: Request):
    ws = _ws(request)
    target = datasets_service.resolve_in_root(ws, body.source.root_id, body.source.path)
    return datasets_service.import_dataset(
        ws, target, dataset_id=body.dataset_id, display_name=body.display_name, signal=body.signal,
        origin=body.origin, ratios=body.split.ratios, guard_samples=body.split.guard_samples,
        csv_options=body.options, expected_sha256=body.expected_sha256,
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetManifest, tags=["datasets"],
            dependencies=[Depends(require_session)])
def dataset_get(dataset_id: str, request: Request):
    return _ws(request).get_dataset(dataset_id)


@router.get("/datasets/{dataset_id}/analysis", response_model=DatasetAnalysis, tags=["datasets"],
            dependencies=[Depends(require_session)])
def dataset_analysis(dataset_id: str, request: Request, version: str = Query("raw-v1", max_length=64)):
    return analyze_dataset(_ws(request), dataset_id, version)


@router.post("/datasets/{dataset_id}/manifest", response_model=DatasetManifest, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_update(dataset_id: str, body: ManifestUpdate, request: Request):
    return datasets_service.update_manifest(_ws(request), dataset_id, signal=body.signal, display_name=body.display_name,
                                            origin=body.origin, notes=body.notes)


@router.get("/datasets/{dataset_id}/diagnostics", response_model=Optional[DiagnosticReport], tags=["datasets"],
            dependencies=[Depends(require_session)])
def dataset_diagnostics_latest(dataset_id: str, request: Request):
    _ws(request).get_dataset(dataset_id)
    return datasets_service.latest_report(_ws(request), dataset_id)


@router.post("/datasets/{dataset_id}/diagnostics", response_model=DiagnosticReport, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_diagnostics_run(dataset_id: str, request: Request, version: str = Query("raw-v1", max_length=64)):
    return datasets_service.run_doctor(_ws(request), dataset_id, version)


@router.post("/datasets/{dataset_id}/preprocess/preview", response_model=PreprocessPreview, tags=["datasets"],
             dependencies=[Depends(require_session)])
def dataset_preprocess_preview(dataset_id: str, body: PreprocessRequest, request: Request):
    return datasets_service.preview_preprocess(_ws(request), dataset_id, body.params, body.base_version)


@router.post("/datasets/{dataset_id}/preprocess", response_model=DatasetVersion, status_code=201, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_preprocess(dataset_id: str, body: PreprocessRequest, request: Request):
    if not body.version:
        raise _error(422, "invalid_request", "a version name is required to create a data version",
                     details=[{"field": "version", "message": "required"}])
    return datasets_service.create_version(_ws(request), dataset_id, body.version, body.params, body.base_version)


@router.post("/datasets/import-builtin", response_model=DatasetManifest, status_code=201, tags=["datasets"],
             dependencies=[Depends(require_csrf)])
def dataset_import_builtin(body: ImportBuiltinRequest, request: Request):
    return _ws(request).register_builtin_dataset(body.name, dataset_id=body.dataset_id)


# --- experiments and runs ------------------------------------------------------------

class ValidateRequest(BaseModel):
    config: Dict[str, Any]


@router.post("/experiments/validate", tags=["experiments"], dependencies=[Depends(require_session)])
def experiments_validate(body: ValidateRequest, request: Request) -> Dict[str, Any]:
    """Resolve and bind without starting anything; config problems are data, not 4xx."""
    return experiments.validate_experiment(_ws(request), body.config).to_dict()


class SubmitRunRequest(BaseModel):
    config: ExperimentConfig
    name: Optional[str] = None
    idempotency_key: Optional[str] = Field(default=None, max_length=256)


class RunView(RunRecord):
    heartbeat_stale: bool = False


def _view(record: RunRecord) -> RunView:
    return RunView(**record.model_dump(), heartbeat_stale=heartbeat_is_stale(record, utcnow(), HEARTBEAT_TIMEOUT))


class RunCount(BaseModel):
    count: int


@router.get("/runs", response_model=List[RunView], tags=["runs"], dependencies=[Depends(require_session)])
def runs_list(request: Request, status: Optional[RunStatus] = None, limit: int = Query(50, ge=1, le=500),
              offset: int = Query(0, ge=0), q: Optional[str] = Query(None, max_length=200)):
    """One page of runs, newest first; ``q`` is a case-insensitive substring search over id, name, dataset and model."""
    request.app.state.supervisor.index_workspace()       # runs made by the CLI / Python API meanwhile
    return [_view(r) for r in request.app.state.store.list_runs(status=status, limit=limit, offset=offset, q=q)]


@router.get("/runs/count", response_model=RunCount, tags=["runs"], dependencies=[Depends(require_session)])
def runs_count(request: Request, status: Optional[RunStatus] = None, q: Optional[str] = Query(None, max_length=200)):
    """How many runs match the same filters as the listing (for paging)."""
    request.app.state.supervisor.index_workspace()
    return RunCount(count=request.app.state.store.count_runs(status=status, q=q))


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
    if record is None and request.app.state.supervisor.index_workspace(run_id):
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


@router.get("/runs/{run_id}/history", response_model=List[HistoryPoint], tags=["runs"],
            dependencies=[Depends(require_session)])
def runs_history(run_id: str, request: Request):
    """Per-epoch validation / test metrics from the run's history log (the same shape as `metric` events)."""
    _get_run(request, run_id)
    try:
        return experiments.training_history(request.app.state.ws, run_id)
    except WorkspaceError as err:
        raise _error(404, "history_not_available", str(err))


@router.get("/runs/{run_id}/live", response_model=Dict[str, Any], tags=["runs"],
            dependencies=[Depends(require_session)])
def runs_live(run_id: str, request: Request):
    """Latest bounded display snapshot; the worker alone computes the plots."""
    from opendpd.services.live import load_live
    _get_run(request, run_id)
    return load_live(request.app.state.ws, run_id)


@router.get("/runs/{run_id}/lineage", response_model=RunLineage, tags=["runs"],
            dependencies=[Depends(require_session)])
def runs_lineage(run_id: str, request: Request):
    """The experiment graph around a run: PA surrogate, DPD model and retry links with the checkpoint hashes used."""
    _get_run(request, run_id)
    return experiments.lineage(request.app.state.ws, run_id)


@router.get("/metrics/profiles", response_model=List[MetricProfile], tags=["metrics"], dependencies=[Depends(require_session)])
def metric_profiles():
    """Every registered metric profile: the definition behind each score."""
    return list_profiles()


@router.get("/metrics/profiles/{profile_id}", response_model=MetricProfile, tags=["metrics"],
            dependencies=[Depends(require_session)])
def metric_profile(profile_id: str):
    try:
        return get_profile(profile_id)
    except KeyError as err:
        raise _error(404, "profile_not_found", str(err))


@router.get("/results/compare", tags=["results"], dependencies=[Depends(require_session)],
            responses={200: {"model": ComparisonReport}})
def results_compare(request: Request, runs: List[str] = Query(..., min_length=2, max_length=8),
                    profile: Optional[str] = Query(None, max_length=64),
                    format: Literal["json", "csv"] = Query("json")):
    """Stored results side by side with the protocol differences that forbid ranking. `format=csv` returns the
    same numbers with their provenance rows; nothing is recomputed."""
    for run_id in runs:
        _get_run(request, run_id)
    try:
        report = compare_results(request.app.state.ws, runs, profile)
    except WorkspaceError as err:
        raise _error(404, "result_not_available", str(err),
                     hint="results exist only for succeeded train_pa / train_dpd / run_dpd runs")
    if format == "csv":
        return Response(content=comparison_csv(report), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=\"comparison.csv\""})
    return report


@router.get("/results/{run_id}/profiles", response_model=List[str], tags=["results"],
            dependencies=[Depends(require_session)])
def results_profiles(run_id: str, request: Request):
    """Metric profiles under which this run has a stored result (primary first)."""
    _get_run(request, run_id)
    return available_profiles(request.app.state.ws, run_id)


@router.get("/results/{run_id}", response_model=EvaluationResult, tags=["results"],
            dependencies=[Depends(require_session)])
def results_get(run_id: str, request: Request, profile: Optional[str] = Query(None, max_length=64)):
    record = _get_run(request, run_id)
    result = experiments.load_result(request.app.state.ws, run_id, profile)
    if result is None:
        stored = available_profiles(request.app.state.ws, run_id)
        raise _error(404, "result_not_available",
                     f"run '{run_id}' has no formal result" + (f" under profile '{profile}'" if profile else "")
                     + f" (status {record.status.value})",
                     hint=("stored profiles: " + ", ".join(stored)) if stored
                     else "results exist only for succeeded train_pa / train_dpd runs")
    return result


# --- adaptation reports (conditions-v1, plan S17) -------------------------------------------

@router.get("/adaptation/reports", response_model=List[AdaptationReportSummary], tags=["adaptation"],
            dependencies=[Depends(require_session)])
def adaptation_reports(request: Request):
    """Hash-bound adaptation reports stored under <workspace>/adaptation/ (built by `opendpd adaptation report`)."""
    return [r.summary() for r in adaptation_service.list_reports(request.app.state.ws)]


@router.get("/adaptation/reports/{plan_sha}", tags=["adaptation"], dependencies=[Depends(require_session)],
            responses={200: {"model": AdaptationReport}})
def adaptation_report(plan_sha: str, request: Request, format: Literal["json", "md"] = Query("json")):
    """One report by its plan hash (the first 12 characters suffice); `format=md` downloads the Markdown rendering."""
    if not re.fullmatch(r"[0-9a-f]{12,64}", plan_sha):
        raise _error(404, "report_not_found", "a plan hash is 12 to 64 hex characters")
    report = next((r for r in adaptation_service.list_reports(request.app.state.ws) if r.plan_sha256.startswith(plan_sha)), None)
    if report is None:
        raise _error(404, "report_not_found", f"no adaptation report for plan '{plan_sha}'",
                     hint="build one with `opendpd adaptation report <plan.json> --workspace <ws>`")
    if format == "md":
        return Response(content=adaptation_service.report_markdown(report), media_type="text/markdown",
                        headers={"Content-Disposition": f"attachment; filename=\"adaptation-{report.plan_sha256[:12]}.md\""})
    return report


# --- reports, packages ----------------------------------------------------------------------

@router.get("/results/{run_id}/report", tags=["results"], dependencies=[Depends(require_session)])
def results_report(run_id: str, request: Request, format: Literal["html", "md"] = Query("html"), language: Optional[UILanguage] = Query(None)):
    """A report bound to the stored result and plot data (nothing recomputed), for download."""
    _get_run(request, run_id)
    ws = request.app.state.ws
    try:
        if format == "md":
            body, media, ext = report_markdown(ws, run_id, language=language or ws.settings().language or "en"), "text/markdown", "md"
        else:
            body, media, ext = report_html(ws, run_id, language=language or ws.settings().language or "en"), "text/html", "html"
    except (WorkspaceError, FileNotFoundError) as err:
        raise _error(404, "report_not_available", str(err))
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition": f"attachment; filename=\"{run_id}-report.{ext}\""})


class ExportRequest(BaseModel):
    run_id: str = Field(max_length=128)
    kind: Literal["full", "share"] = "share"


class ExportInfo(BaseModel):
    export_id: str
    filename: str
    size_bytes: int
    download_url: str
    manifest: PackageManifest


_EXPORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


def _package_error(err: PackageError) -> HTTPException:
    return _error(413 if err.code == "too_large" else 422, err.code, str(err), hint=err.hint)


@router.post("/exports", response_model=ExportInfo, status_code=201, tags=["exports"],
             dependencies=[Depends(require_csrf)])
def exports_create(body: ExportRequest, request: Request, language: Optional[UILanguage] = Query(None)):
    """Write an experiment package into <workspace>/exports. `share` leaves user data, logs and machine paths out."""
    record = _get_run(request, body.run_id)
    if record.status != RunStatus.succeeded:
        raise _error(409, "run_not_finished", "only succeeded runs can be exported")
    ws = request.app.state.ws
    stamp = utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{body.run_id}-{body.kind}-{stamp}.zip"
    try:
        manifest = packages.export_run(ws, body.run_id, ws.exports_dir / filename, kind=body.kind, language=language or ws.settings().language or "en")
    except PackageError as err:
        raise _package_error(err)
    except (WorkspaceError, FileNotFoundError) as err:
        raise _error(409, "export_failed", str(err))
    export_id = filename[:-4]
    return ExportInfo(export_id=export_id, filename=filename, size_bytes=(ws.exports_dir / filename).stat().st_size,
                      download_url=f"/api/v1/exports/{export_id}", manifest=manifest)


# --- deployment packages (fixed-point-v1, plan S19) ------------------------------------------

class DeployRequest(BaseModel):
    run_id: str = Field(max_length=128)


class DeployExportInfo(BaseModel):
    export_id: str
    filename: str
    size_bytes: int
    download_url: str
    manifest: DeploymentManifest


@router.post("/deploy/exports", response_model=DeployExportInfo, status_code=201, tags=["exports"],
             dependencies=[Depends(require_csrf)])
def deploy_export(body: DeployRequest, request: Request):
    """Quantise a finished GRU run under fixed-point-v1, write golden vectors and the C99 reference, verify it bit for
    bit, and serve the package from <workspace>/exports. Unsupported models are refused with the reason."""
    record = _get_run(request, body.run_id)
    if record.status != RunStatus.succeeded:
        raise _error(409, "run_not_finished", "only succeeded runs can be exported")
    reason = deploy_service.support(record.model_key or "")
    if reason:
        raise _error(422, "deploy_unsupported", reason, hint="models with export_formats containing fixed-point-v1 (see GET /models)")
    ws = request.app.state.ws
    filename = f"{body.run_id}-deploy-{utcnow().strftime('%Y%m%d-%H%M%S')}.zip"
    try:
        manifest = deploy_service.export_deployment(ws, body.run_id, ws.exports_dir / filename)
    except (WorkspaceError, ValueError, RuntimeError) as err:
        raise _error(422, "deploy_failed", str(err))
    path = ws.exports_dir / filename
    return DeployExportInfo(export_id=path.stem, filename=filename, size_bytes=path.stat().st_size,
                            download_url=f"/api/v1/exports/{path.stem}", manifest=manifest)


@router.get("/exports/{export_id}", tags=["exports"], dependencies=[Depends(require_session)])
def exports_download(export_id: str, request: Request):
    if not _EXPORT_ID.match(export_id):
        raise _error(404, "export_not_found", "no such export")
    path = request.app.state.ws.exports_dir / f"{export_id}.zip"
    if not path.is_file():
        raise _error(404, "export_not_found", f"export '{export_id}' does not exist")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.post("/imports", response_model=ImportReport, status_code=201, tags=["exports"],
             dependencies=[Depends(require_csrf)])
async def imports_create(request: Request, file: UploadFile):
    """Upload an experiment package and import it; every hash is verified before anything is written."""
    ws = _ws(request)

    def chunks():
        while True:
            chunk = file.file.read(1 << 20)
            if not chunk:
                return
            yield chunk

    try:
        path = packages.receive_package(ws, file.filename or "package.zip", chunks(), UPLOAD_MAX_BODY)
        return packages.import_package(ws, path)
    except PackageError as err:
        raise _package_error(err)
    except WorkspaceError as err:
        raise _error(409, "import_failed", str(err))


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
def runs_logs(run_id: str, request: Request, offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=2000),
              tail: bool = False):
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
        if tail and offset == 0:
            # Read a bounded suffix, then keep its last `limit` complete lines.
            f.seek(max(0, size - 262144))
            if f.tell():
                f.readline(size - f.tell())   # discard the possibly partial first line
            start = f.tell()
            raw_lines = f.read(max(0, size - start)).splitlines(keepends=True)
            if raw_lines and not raw_lines[-1].endswith(b"\n") and not finished:
                raw_lines.pop()
            skip = max(0, len(raw_lines) - limit)
            pos = start + sum(map(len, raw_lines[:skip]))
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
