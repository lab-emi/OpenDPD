"""Experiment configuration: the single vocabulary for GUI, CLI and API.

``ExperimentConfig`` is what a user writes or the GUI submits; it may omit
defaults. ``ResolvedExperimentConfig`` is what actually ran: every default
that influences a result is filled in and the whole thing is hashed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Literal, Optional, Union

from pydantic import Field, model_validator

from .measurement import MeasurementConfig
from .common import SCHEMA_VERSION, EvidenceType, Sha256, Slug, StrictModel, utcnow

ParamValue = Union[int, float, str, bool]


class TaskType(str, Enum):
    train_pa = "train_pa"
    train_dpd = "train_dpd"
    run_dpd = "run_dpd"
    evaluate_measured = "evaluate_measured"   # score captures of a physical PA driven by a run_dpd export (S16)


class DatasetRef(StrictModel):
    id: Slug
    preprocessing_version: str = "raw-v1"
    split_version: str = "contiguous-v1"


class ModelSpec(StrictModel):
    key: Slug                                          # registry key, e.g. "gru"
    parameters: Dict[str, ParamValue] = Field(default_factory=dict)


class TrainingConfig(StrictModel):
    """Defaults mirror the OpenDPDv2 recipe (arguments.py); they are the
    single source of defaults for every entry point."""

    epochs: int = Field(default=300, ge=1)
    batch_size: int = Field(default=64, ge=1)
    batch_size_eval: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=5e-3, gt=0)
    lr_end: float = Field(default=5e-5, gt=0)
    lr_schedule: bool = True
    decay_factor: float = Field(default=0.5, gt=0, lt=1)
    patience: int = Field(default=5, ge=0)
    optimizer: Literal["adamw", "adam", "sgd", "rmsprop"] = "adamw"
    loss: Literal["l2", "l1"] = "l2"
    grad_clip: float = Field(default=200.0, ge=0)
    frame_length: int = Field(default=200, ge=1)
    frame_stride: int = Field(default=1, ge=1)
    seed: int = Field(default=0, ge=0)
    reproducibility: Literal["soft", "hard"] = "soft"
    eval_val: bool = True
    eval_test: bool = True

    @model_validator(mode="after")
    def _lr_bounds(self) -> "TrainingConfig":
        if self.lr_end > self.learning_rate:
            raise ValueError("lr_end must not exceed learning_rate")
        return self


class EvaluationConfig(StrictModel):
    profile_id: Slug = "legacy-opendpd-v1"
    # None = derived from the task (train_pa -> pa_modeling, train_dpd/run_dpd -> dpd_surrogate);
    # a value that contradicts the task is rejected.
    evidence_type: Optional[EvidenceType] = None
    # Which *validation* metric selects the checkpoint. Filled by resolution.
    checkpoint_selection_metric: Optional[str] = None


class ExecutionConfig(StrictModel):
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    device_index: int = Field(default=0, ge=0)
    num_threads: Optional[int] = Field(default=None, ge=1)
    cuda_graph_training: bool = False


class QuantizationConfig(StrictModel):
    enabled: bool = False
    n_bits_w: int = Field(default=8, ge=2, le=32)
    n_bits_a: int = Field(default=8, ge=2, le=32)
    pretrained_run_id: Optional[Slug] = None
    label: str = ""


class PAReference(StrictModel):
    """The frozen PA surrogate a DPD task trains or evaluates through."""

    run_id: Slug
    checkpoint_artifact_id: Optional[Slug] = None
    checkpoint_sha256: Optional[Sha256] = None
    model: Optional[ModelSpec] = None


class DPDReference(StrictModel):
    run_id: Slug
    checkpoint_artifact_id: Optional[Slug] = None
    checkpoint_sha256: Optional[Sha256] = None
    model: Optional[ModelSpec] = None


class ExperimentConfig(StrictModel):
    schema_version: int = SCHEMA_VERSION
    task: TaskType
    recipe_id: Optional[Slug] = None
    name: Optional[str] = None
    dataset: DatasetRef
    model: ModelSpec
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    pa_reference: Optional[PAReference] = None
    dpd_reference: Optional[DPDReference] = None
    quantization: Optional[QuantizationConfig] = None
    measurement: Optional[MeasurementConfig] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _task_rules(self) -> "ExperimentConfig":
        t = self.task
        if self.evaluation.evidence_type is None:
            derived = {TaskType.train_pa: EvidenceType.pa_modeling,
                       TaskType.evaluate_measured: EvidenceType.dpd_measured}.get(t, EvidenceType.dpd_surrogate)
            self.evaluation = self.evaluation.model_copy(update={"evidence_type": derived})
        ev = self.evaluation.evidence_type
        if self.measurement is not None and t != TaskType.evaluate_measured:
            raise ValueError("only evaluate_measured takes a measurement block")
        if t == TaskType.train_pa:
            if ev != EvidenceType.pa_modeling:
                raise ValueError("train_pa produces pa_modeling evidence")
            if self.pa_reference is not None or self.dpd_reference is not None:
                raise ValueError("train_pa takes no PA/DPD reference")
        elif t == TaskType.train_dpd:
            if ev != EvidenceType.dpd_surrogate:
                raise ValueError("train_dpd through a PA surrogate produces dpd_surrogate evidence")
            if self.pa_reference is None:
                raise ValueError("train_dpd requires pa_reference (a compatible PA surrogate run)")
        elif t == TaskType.run_dpd:
            if self.dpd_reference is None:
                raise ValueError("run_dpd requires dpd_reference (the trained DPD run)")
        elif t == TaskType.evaluate_measured:
            if ev != EvidenceType.dpd_measured:
                raise ValueError("evaluate_measured scores a physical PA and produces dpd_measured evidence")
            if self.measurement is None:
                raise ValueError("evaluate_measured requires a measurement block (played run, captures, conditions)")
            if self.pa_reference is not None:
                raise ValueError("evaluate_measured takes no PA surrogate: the PA output is captured, not simulated")
        return self


class ResolutionInfo(StrictModel):
    resolver_version: str
    resolved_at: datetime = Field(default_factory=utcnow)
    defaults_source: str                       # e.g. "opendpd 2.2.0 / schema 1 / registry gru v1"
    config_sha256: Sha256                      # hash of the canonical resolved JSON (excluding this block)
    warnings: list[str] = Field(default_factory=list)


class ResolvedExperimentConfig(ExperimentConfig):
    """Same shape as ``ExperimentConfig`` but every result-affecting default is
    explicit, model parameters are complete, and the checkpoint selection
    metric is set. Produced only by ``opendpd.services.config.resolve``."""

    resolution: ResolutionInfo

    @model_validator(mode="after")
    def _fully_resolved(self) -> "ResolvedExperimentConfig":
        if self.evaluation.checkpoint_selection_metric is None:
            raise ValueError("resolved config must state checkpoint_selection_metric")
        if self.task == TaskType.train_dpd and self.pa_reference is not None:
            if self.pa_reference.model is None or self.pa_reference.checkpoint_sha256 is None:
                raise ValueError("resolved DPD config must bind the PA surrogate model and checkpoint hash")
        return self
