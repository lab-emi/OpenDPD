"""Versioned data contracts shared by the GUI, CLI and Python API.

Everything the three entry points exchange (dataset manifests, diagnostic
reports, experiment configurations, run records and events, metric profiles,
evaluation results and artifact manifests) is defined here once, with
Pydantic v2. The FastAPI OpenAPI document and the frontend types are
generated from these models; nothing is hand-written twice.

Scientific semantics are versioned: ``SCHEMA_VERSION`` covers the field
layout, ``MetricProfile.version`` covers metric definitions.
"""

from .artifacts import Artifact, ArtifactKind, ArtifactManifest
from .common import (
    SCHEMA_VERSION,
    BetterDirection,
    EvidenceType,
    FileRef,
    MetricStatus,
    MetricValue,
    Severity,
    SoftwareProvenance,
    utcnow,
)
from .dataset import (
    DatasetManifest,
    DatasetVersion,
    PreprocessingParams,
    DatasetOrigin,
    DatasetSource,
    DatasetSourceKind,
    SignalSpec,
    SplitSpec,
)
from .diagnostics import DiagnosticItem, DiagnosticReport
from .experiment import (
    DatasetRef,
    DPDReference,
    InitReference,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelSpec,
    PAReference,
    QuantizationConfig,
    ResolvedExperimentConfig,
    ResolutionInfo,
    TaskType,
    TrainingConfig,
)
from .fixed_point import (
    DeploymentManifest,
    FixedPointReport,
    FixedPointSpec,
    GoldenCase,
    MetricDelta,
    Verification,
)
from .leaderboard import (
    BoardEntry,
    CheckItem,
    ComparabilityKey,
    DataCard,
    HistoryEvent,
    Leaderboard,
    LicenceCheck,
    MethodCard,
    ModelCard,
    PackageRef,
    Recomputation,
    ResourceBudget,
    ResultSummary,
    Review,
    SubmissionCard,
    SubmissionCheck,
    Submitter,
)
from .conditions import (
    AdaptationCell,
    AdaptationEntry,
    AdaptationPlan,
    AdaptationReport,
    AdaptationReportSummary,
    AdaptationTask,
    CellAggregate,
    Condition,
    ConditionAudit,
    ConditionSet,
    EvidenceBar,
    TargetRule,
)
from .measurement import (
    ATTESTATION,
    MOCK_ATTESTATION,
    CaptureAlignment,
    CaptureRef,
    MeasurementConditions,
    MeasurementConfig,
    MeasurementEvidence,
)
from .metrics import MetricDefinition, MetricProfile, ProfileValidation
from .waveform import WaveformBinding, WaveformSpec
from .benchmark import (
    BaselineBand,
    BenchmarkEntry,
    BenchmarkPlan,
    BenchmarkReport,
    DataAudit,
    EntryResult,
    MetricStats,
    RegressionBaseline,
    RegressionCheck,
    RegressionItem,
    SeedScore,
)
from .package import (
    PACKAGE_VERSION,
    ImportReport,
    PackageDataset,
    PackageFile,
    PackageManifest,
    PackageReference,
)
from .results import (
    BaselineScore,
    ComparisonPair,
    ComparisonReport,
    DatasetEvidence,
    EvaluationResult,
    ExecutionEvidence,
    StreamConsistency,
    HistoryPoint,
    ModelEvidence,
    ScalingInfo,
    SignalReference,
    SignalStage,
    SurrogateCoverage,
)
from .run import (
    TERMINAL_STATUSES,
    LineageLink,
    LineageRelation,
    RunError,
    RunEvent,
    RunEventType,
    RunRecord,
    RunLineage,
    RunStatus,
    WorkerInfo,
    can_transition,
    heartbeat_is_stale,
)

from .settings import UI_LANGUAGES, UILanguage, WorkspaceSettings

__all__ = [
    "UI_LANGUAGES",
    "UILanguage",
    "WorkspaceSettings",
    "BaselineBand", "BenchmarkEntry", "BenchmarkPlan", "BenchmarkReport", "DataAudit",
    "EntryResult", "MetricStats", "RegressionBaseline", "RegressionCheck", "RegressionItem", "SeedScore",
    "SCHEMA_VERSION", "utcnow",
    "Artifact", "ArtifactKind", "ArtifactManifest",
    "BetterDirection", "EvidenceType", "FileRef", "MetricStatus", "MetricValue", "Severity",
    "SoftwareProvenance",
    "DatasetManifest",
    "DatasetVersion",
    "PreprocessingParams", "DatasetOrigin", "DatasetSource", "DatasetSourceKind", "SignalSpec", "SplitSpec",
    "DiagnosticItem", "DiagnosticReport",
    "DatasetRef", "DPDReference", "InitReference", "EvaluationConfig", "ExecutionConfig", "ExperimentConfig",
    "ModelSpec", "PAReference", "QuantizationConfig", "ResolvedExperimentConfig", "ResolutionInfo",
    "TaskType", "TrainingConfig",
    "MetricDefinition", "MetricProfile", "ProfileValidation", "WaveformBinding", "WaveformSpec",
    "ATTESTATION", "MOCK_ATTESTATION", "CaptureAlignment", "CaptureRef", "MeasurementConditions",
    "MeasurementConfig", "MeasurementEvidence",
    "AdaptationCell", "AdaptationEntry", "AdaptationPlan", "AdaptationReport", "AdaptationReportSummary", "AdaptationTask",
    "CellAggregate",
    "BoardEntry", "CheckItem", "ComparabilityKey", "DataCard", "HistoryEvent", "Leaderboard", "LicenceCheck", "MethodCard",
    "ModelCard", "PackageRef", "Recomputation", "ResourceBudget", "ResultSummary", "Review", "SubmissionCard", "SubmissionCheck",
    "Submitter",
    "Condition", "ConditionAudit", "ConditionSet", "EvidenceBar", "TargetRule",
    "PACKAGE_VERSION", "ImportReport", "PackageDataset", "PackageFile", "PackageManifest", "PackageReference",
    "BaselineScore", "ComparisonPair", "ComparisonReport", "DatasetEvidence", "EvaluationResult", "ExecutionEvidence", "StreamConsistency",
    "DeploymentManifest", "FixedPointReport", "FixedPointSpec", "GoldenCase", "MetricDelta", "Verification", "HistoryPoint",
    "ModelEvidence", "ScalingInfo", "SignalReference", "SignalStage", "SurrogateCoverage",
    "TERMINAL_STATUSES", "LineageLink", "LineageRelation", "RunError", "RunEvent", "RunEventType", "RunLineage",
    "RunRecord", "RunStatus", "WorkerInfo", "can_transition", "heartbeat_is_stale",
]
