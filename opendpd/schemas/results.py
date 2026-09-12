"""Formal evaluation results: every score carries its evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from .common import (
    SCHEMA_VERSION,
    EvidenceType,
    FileRef,
    MetricValue,
    Sha256,
    Slug,
    SoftwareProvenance,
    StrictModel,
    utcnow,
)
from .experiment import ModelSpec, ParamValue
from .measurement import MeasurementEvidence


class SignalReference(StrictModel):
    """What the evaluated signal was compared against."""

    kind: Literal["linear_gain_target", "measured_pa_output", "pa_surrogate_output"]
    description: str = Field(min_length=1)
    gain_rule: Optional[str] = None      # e.g. "max|y_train| / max|x_train| (legacy set_target_gain)"
    gain_value: Optional[float] = None


class SignalStage(StrictModel):
    """One link of the evaluated signal chain: ``x`` (target input), ``u = DPD(x)``
    (pre-distorted PA input) and ``y`` (PA output). ``simulated`` is True when a
    learned surrogate produced the signal; it is never inferred from the task."""

    symbol: Literal["x", "u", "y"]
    role: str = Field(min_length=1)
    source: str = Field(min_length=1)
    simulated: bool = False
    n_samples: Optional[int] = Field(default=None, ge=0)
    peak_abs: Optional[float] = Field(default=None, ge=0)
    rms: Optional[float] = Field(default=None, ge=0)
    artifact_id: Optional[Slug] = None      # exported file in the run's artifact manifest, when any


class BaselineScore(StrictModel):
    """Scores of a comparison signal under the *same* reference, profile and
    valid range as ``EvaluationResult.metrics``; never normalised separately.
    A measured baseline is a second capture: it carries its own alignment (delay
    and least-squares gain, both recorded in the measurement) and the level
    difference between the captures is reported, never scaled away."""

    kind: Literal["surrogate_without_dpd", "measured_without_dpd"]
    description: str = Field(min_length=1)
    metrics: List[MetricValue] = Field(min_length=1)


class SurrogateCoverage(StrictModel):
    """How far the pre-distorted signal leaves the amplitude range the PA
    surrogate was fitted on. Evidence for an extrapolation warning only: staying
    inside the range does not prove the surrogate valid."""

    fitted_peak_abs: float = Field(ge=0)              # max |x| over the surrogate's training split
    u_peak_abs: float = Field(ge=0)
    fraction_above_fitted_peak: float = Field(ge=0, le=1)
    note: str = Field(min_length=1)


class ScalingInfo(StrictModel):
    """What the amplitudes are relative to. Without a physical calibration no
    absolute power (dBm) or efficiency is derived from them."""

    amplitude_units: Literal["normalized", "volts", "unknown"]
    input_scaling: str = Field(min_length=1)
    reference_gain: Optional[float] = None
    physical_calibration: bool = False


class StreamConsistency(StrictModel):
    """Max |streamed - full sequence| of the same variant run from the same reset, over the valid range (streaming-v1)."""

    chunk_samples: int = Field(ge=1)
    max_abs_error: float = Field(ge=0)
    tolerance: float = Field(gt=0)
    within_tolerance: bool


class ExecutionEvidence(StrictModel):
    """How the evaluated model consumed the signal (plan S18). Present for streaming variants only; offline
    results are segment-wise from a zero state and say so through ``ModelEvidence.execution_semantics``."""

    semantics: str = Field(min_length=1)
    state: Literal["recurrent", "window"]
    chunk_samples: int = Field(ge=1)
    lookahead_samples: int = Field(ge=0)
    # lookahead_samples / sample rate: the future an output needs (an information bound), not a measured latency
    lookahead_s: Optional[float] = Field(default=None, ge=0)
    history_samples: Optional[int] = Field(default=None, ge=0)
    warmup_samples: Optional[int] = Field(default=None, ge=0)  # measured on this signal (None = not measured)
    consistency: StreamConsistency                             # the tail is zero-padded at flush (streaming-v1)


class DatasetEvidence(StrictModel):
    dataset_id: Slug
    split: Literal["train", "val", "test"]
    raw_sha256: Optional[Sha256] = None
    processed_sha256: Optional[Sha256] = None
    preprocessing_version: str
    split_version: str
    n_samples: Optional[int] = Field(default=None, ge=0)


class ModelEvidence(StrictModel):
    role: Literal["pa", "dpd"]
    model: ModelSpec
    run_id: Optional[Slug] = None
    weights_sha256: Optional[Sha256] = None
    n_parameters: Optional[int] = Field(default=None, ge=0)
    execution_semantics: str = "offline_segmented"   # vs. "streaming"
    lookahead_samples: Optional[int] = Field(default=None, ge=0)
    # how the weights were obtained: gradient descent (PA), gradient descent through the surrogate (DPD, DLA),
    # direct least squares (PA) or indirect learning by least squares on measured data (DPD, ILA)
    training_path: Optional[Literal["gradient", "gradient_dla", "least_squares", "ila_least_squares"]] = None


class EvaluationResult(StrictModel):
    schema_version: int = SCHEMA_VERSION
    result_id: Slug
    run_id: Optional[Slug] = None
    generated_at: datetime = Field(default_factory=utcnow)
    source: Literal["opendpd-studio", "legacy-log-import", "mock"]
    is_mock: bool = False
    evidence_type: EvidenceType
    metric_profile_id: Slug
    metric_profile_version: int = Field(ge=1)
    dataset: DatasetEvidence
    models: List[ModelEvidence] = Field(default_factory=list)
    reference: SignalReference
    valid_sample_range: Optional[Tuple[int, int]] = None
    n_segments: Optional[int] = Field(default=None, ge=0)
    nperseg: Optional[int] = Field(default=None, ge=2)
    metrics: List[MetricValue] = Field(min_length=1)
    selected_epoch: Optional[int] = Field(default=None, ge=0)
    history: Optional[FileRef] = None
    software: SoftwareProvenance
    device: str = Field(min_length=1)
    seed: Optional[int] = None
    numeric_mode: str = "float32"
    limitations: List[str] = Field(default_factory=list)
    signal_chain: List[SignalStage] = Field(default_factory=list)
    baselines: List[BaselineScore] = Field(default_factory=list)
    surrogate_coverage: Optional[SurrogateCoverage] = None
    scaling: Optional[ScalingInfo] = None
    measurement: Optional[MeasurementEvidence] = None   # dpd_measured only: conditions, captures, alignment
    execution: Optional[ExecutionEvidence] = None       # streaming variants only (S18)
    extra: Dict[str, ParamValue] = Field(default_factory=dict)

    def metric(self, name: str) -> MetricValue:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)

    @model_validator(mode="after")
    def _evidence_rules(self) -> "EvaluationResult":
        if self.is_mock != (self.source == "mock"):
            raise ValueError("is_mock must be true exactly when source is 'mock'")
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("duplicate metric names in result")
        roles = {m.role for m in self.models}
        if self.evidence_type == EvidenceType.pa_modeling:
            if self.reference.kind != "measured_pa_output":
                raise ValueError("pa_modeling evidence compares against measured PA output")
            if "pa" not in roles:
                raise ValueError("pa_modeling result must name the PA model")
        elif self.evidence_type == EvidenceType.dpd_surrogate:
            if "pa" not in roles or "dpd" not in roles:
                raise ValueError("dpd_surrogate result must name both the DPD and the PA surrogate")
            if self.reference.kind != "linear_gain_target":
                raise ValueError("dpd_surrogate evidence compares the cascade against the linear gain target")
            if not any("surrogate" in lim.lower() for lim in self.limitations):
                raise ValueError("dpd_surrogate results must state the surrogate limitation")
        elif self.evidence_type == EvidenceType.dpd_measured:
            if "dpd" not in roles:
                raise ValueError("dpd_measured result must name the DPD model")
            if self.reference.kind == "pa_surrogate_output":
                raise ValueError("dpd_measured evidence cannot come from a PA surrogate")
            if self.source == "opendpd-studio" and self.measurement is None:
                raise ValueError("a dpd_measured result produced here records its measurement (conditions and captures)")
        if self.measurement is not None and self.evidence_type != EvidenceType.dpd_measured:
            raise ValueError("only dpd_measured results carry a measurement record")
        if self.source == "legacy-log-import" and not self.limitations:
            raise ValueError("legacy imports must list what is unknown")
        symbols = [s.symbol for s in self.signal_chain]
        if len(symbols) != len(set(symbols)):
            raise ValueError("each signal chain stage appears once")
        for stage in self.signal_chain:
            if stage.symbol == "y" and stage.simulated != (self.evidence_type == EvidenceType.dpd_surrogate):
                raise ValueError("the PA output stage is simulated exactly for dpd_surrogate evidence")
        if self.baselines:
            names = {m.name for m in self.metrics}
            for baseline in self.baselines:
                if {m.name for m in baseline.metrics} != names:
                    raise ValueError(f"baseline {baseline.kind} must score the same metrics as the result")
        if self.scaling is not None and self.scaling.physical_calibration:
            raise ValueError("physical calibration is not implemented; absolute power cannot be claimed")
        return self


class ComparisonPair(StrictModel):
    a: Slug
    b: Slug
    incompatibilities: List[str] = Field(default_factory=list)


class ComparisonReport(StrictModel):
    """Results shown side by side. ``comparable`` is True only when every pair was produced under the same
    protocol (``opendpd.core.metrics.incompatibilities``); otherwise the reasons are listed and nothing is ranked."""

    schema_version: int = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utcnow)
    results: List[EvaluationResult] = Field(min_length=1)
    pairs: List[ComparisonPair] = Field(default_factory=list)
    comparable: bool
    note: str = Field(min_length=1)


class HistoryPoint(StrictModel):
    """One split's metrics after one epoch, read from the run's history log (same shape as ``metric`` events)."""

    epoch: int = Field(ge=0)
    split: Literal["val", "test"]
    values: Dict[str, float] = Field(default_factory=dict)
    train_loss: Optional[float] = None
