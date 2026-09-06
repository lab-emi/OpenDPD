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


class SignalReference(StrictModel):
    """What the evaluated signal was compared against."""

    kind: Literal["linear_gain_target", "measured_pa_output", "pa_surrogate_output"]
    description: str = Field(min_length=1)
    gain_rule: Optional[str] = None      # e.g. "max|y_train| / max|x_train| (legacy set_target_gain)"
    gain_value: Optional[float] = None


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
        if self.source == "legacy-log-import" and not self.limitations:
            raise ValueError("legacy imports must list what is unknown")
        return self
