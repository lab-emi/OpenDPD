"""Strict request and response contracts for the core HTTP API."""
from __future__ import annotations
from opendpd import __version__
from opendpd.core.splits import DEFAULT_GUARD_SAMPLES
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
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
from opendpd.schemas.analysis import DatasetAnalysis
from opendpd.schemas.importing import BuiltinDatasetInfo, CsvInspection, CsvOptions, DatasetImportDefaults
from opendpd.schemas.common import Slug, Sha256, SLUG_PATTERN as ID_PATTERN, StrictModel
from opendpd.schemas.rf import RFConditions
from opendpd.schemas.review import ReviewContext, FigureSpec, SavedFigure
from opendpd.schemas.measurement_session import MeasurementSessionSpec, MeasurementSession
from opendpd.schemas.system import ServerStatus

SLUG_PATTERN = f"^{ID_PATTERN}$"

class BootstrapRequest(StrictModel):
    token: str

class SessionInfo(StrictModel):
    authenticated: bool
    csrf_token: Optional[str] = None
    version: str = __version__

class DeviceInfo(StrictModel):
    device: str
    detected: bool
    name: Optional[str] = None
    count: int = 0
    tested_models: List[str] = Field(default_factory=list)

class Capabilities(StrictModel):
    version: str
    devices: List[DeviceInfo]
    workspace: str
    note: str
    custom_dataset_imports: bool = False

class ParamSpecInfo(StrictModel):
    name: str
    type: str
    default: Any
    description: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[List[Any]] = None
    legacy_arg: Dict[str, str] = Field(default_factory=dict)

class ModelInfo(StrictModel):
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

class RecipeInfo(StrictModel):
    recipe_id: str
    title: str
    purpose: str
    task: TaskType
    model: ModelSpec
    training: TrainingConfig
    description: str
    limits: str
    expected_duration: str

class ImportBuiltinRequest(StrictModel):
    name: str
    # An identifier, never a path fragment: it names a directory under datasets/.
    dataset_id: Optional[Slug] = None

class ImportRootInfo(StrictModel):
    root_id: str
    path: str
    exists: bool

class FileEntryInfo(StrictModel):
    path: str
    kind: str
    size_bytes: int = 0

class SourceRef(StrictModel):
    """A file inside an authorised import root; clients never send absolute paths."""
    root_id: str
    path: str = Field(max_length=1024)

class SourceInfoOut(StrictModel):
    kind: str
    path: str
    columns: List[str]
    suggested_mapping: Dict[str, str]
    n_rows: Optional[int] = None
    preview: List[Dict[str, float]]
    arrays: Dict[str, Dict[str, Any]]
    problems: List[str]
    legacy_files: List[str]

class ImportRequest(StrictModel):
    source: SourceRef
    dataset_id: Optional[str] = Field(default=None, max_length=64)
    display_name: Optional[str] = Field(default=None, max_length=200)
    mapping: Dict[str, str] = Field(default_factory=dict)
    signal: SignalSpec = SignalSpec()
    origin: DatasetOrigin = DatasetOrigin.unknown
    guard_samples: int = Field(default=DEFAULT_GUARD_SAMPLES, ge=0, le=100_000)
    notes: Optional[str] = Field(default=None, max_length=2000)

class CsvPreviewRequest(StrictModel):
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

class ManifestUpdate(StrictModel):
    signal: Optional[SignalSpec] = None
    display_name: Optional[str] = Field(default=None, max_length=200)
    origin: Optional[DatasetOrigin] = None
    notes: Optional[str] = Field(default=None, max_length=2000)

class PreprocessRequest(StrictModel):
    params: PreprocessingParams
    # Both name a directory under the dataset's versions/, so both are
    # identifiers rather than paths.
    base_version: Slug = "raw-v1"
    version: Optional[Slug] = Field(default=None, max_length=64)

class PreprocessPreview(StrictModel):
    n_samples_before: int
    n_samples_after: int
    record: Dict[str, Any]
    report_after: DiagnosticReport

class UploadValidation(StrictModel):
    status: Literal['passed']
    sha256: Sha256
    n_samples: int
    columns: int

class UploadResult(StrictModel):
    root_id: str
    path: str
    size_bytes: int
    validation: UploadValidation

class ValidateRequest(StrictModel):
    config: Dict[str, Any]

class SubmitRunRequest(StrictModel):
    config: ExperimentConfig
    name: Optional[str] = None
    idempotency_key: Optional[str] = Field(default=None, max_length=256)

class RunView(RunRecord):
    heartbeat_stale: bool = False

class RunCount(StrictModel):
    count: int

class ModelDownloadInfo(StrictModel):
    available: bool
    final: bool
    epoch: Optional[int] = None
    sha256: Optional[Sha256] = None
    size_bytes: Optional[int] = None
    download_url: Optional[str] = None

class ExportRequest(StrictModel):
    run_id: str = Field(max_length=128)
    kind: Literal["full", "share"] = "share"

class ExportInfo(StrictModel):
    export_id: str
    filename: str
    size_bytes: int
    download_url: str
    manifest: PackageManifest

class DeployRequest(StrictModel):
    run_id: str = Field(max_length=128)

class DeployExportInfo(StrictModel):
    export_id: str
    filename: str
    size_bytes: int
    download_url: str
    manifest: DeploymentManifest

class EventPage(StrictModel):
    events: List[RunEvent]
    last_seq: int
    terminal: bool

class LogPage(StrictModel):
    lines: List[str]
    next_offset: int
    eof: bool
    size: int
