"""Studio sweep-v1: bounded, pre-registered work scheduled as ordinary runs."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .common import Sha256, Slug, StrictModel, utcnow
from .conditions import AdaptationPlan, AdaptationTask, ConditionSet
from .experiment import DatasetRef, ExperimentConfig
from .benchmark import MetricStats


class SweepMethod(StrictModel):
    entry_id: Slug
    recipe_id: Optional[Slug] = None
    config: Optional[ExperimentConfig] = None
    pa_entry: Optional[Slug] = None

    @model_validator(mode="after")
    def _one_source(self):
        if (self.recipe_id is None) == (self.config is None):
            raise ValueError("choose a recipe or copy a configuration for each method")
        return self


class SweepDraft(StrictModel):
    protocol_id: Literal["sweep-v1"] = "sweep-v1"
    title: str = Field(min_length=1, max_length=160)
    mode: Literal["same_condition", "cross_condition"]
    dataset: Optional[DatasetRef] = None
    fixed_pa_run_id: Optional[Slug] = None
    condition_set: Optional[ConditionSet] = None
    methods: List[SweepMethod] = Field(min_length=1, max_length=8)
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=20)
    tasks: List[AdaptationTask] = Field(default_factory=lambda: ["zero_update", "few_shot", "full_retrain"], min_length=1, max_length=3)
    budgets: List[int] = Field(default_factory=lambda: [2000], min_length=1, max_length=8)
    metric_profile_id: Slug = "legacy-opendpd-v1"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    max_runs: int = Field(default=64, ge=1, le=256)
    max_wall_clock_seconds: int = Field(default=3600, ge=1, le=604800)

    @model_validator(mode="after")
    def _consistent(self):
        for field in ("seeds", "budgets", "tasks"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must be distinct")
        if any(s < 0 for s in self.seeds) or any(b <= 0 for b in self.budgets):
            raise ValueError("seeds must be nonnegative and sample budgets positive")
        if len({m.entry_id for m in self.methods}) != len(self.methods):
            raise ValueError("method identifiers must be unique")
        if self.mode == "same_condition" and (self.dataset is None or self.condition_set is not None):
            raise ValueError("same-condition sweeps require one dataset and no condition card")
        if self.mode == "cross_condition" and (self.condition_set is None or self.dataset is not None or self.fixed_pa_run_id is not None):
            raise ValueError("cross-condition sweeps require a condition card and train condition-specific PA models")
        if self.condition_set and len(self.condition_set.conditions) > 12:
            raise ValueError("a sweep supports at most twelve conditions")
        return self


class SweepCell(StrictModel):
    cell_id: Slug
    entry_id: Slug
    condition_id: Slug
    task: str
    seed: int
    budget_samples: Optional[int] = None
    train_samples: Optional[int] = None
    epochs: int = Field(ge=0)
    status: Literal["pending", "queued", "running", "cancel_requested", "succeeded", "failed", "cancelled", "interrupted", "blocked"] = "pending"
    run_id: Optional[Slug] = None
    attempts: List[Slug] = Field(default_factory=list)
    reason: Optional[str] = None
    config_sha256: Optional[Sha256] = None


class SweepPreview(StrictModel):
    draft: SweepDraft
    plan_sha256: Sha256
    templates: Dict[str, ExperimentConfig]
    adaptation: Optional[AdaptationPlan] = None
    source_hashes: Dict[str, Sha256]
    cells: List[SweepCell]
    training_runs: int
    evaluation_runs: int
    sample_epochs: Optional[int] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SweepRecord(StrictModel):
    sweep_id: Slug
    preview: SweepPreview
    status: Literal["ready", "running", "complete", "needs_attention", "cancelled", "interrupted"] = "ready"
    cells: List[SweepCell]
    created_at: datetime = Field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    elapsed_seconds: float = Field(default=0, ge=0)
    reason: Optional[str] = None


class SweepStart(StrictModel):
    resume_failed: bool = False


class SweepAggregate(StrictModel):
    entry_id: Slug
    condition_id: Slug
    task: str
    budget_samples: Optional[int] = None
    n_requested_seeds: int
    n_succeeded: int
    metrics: Dict[str, MetricStats]
    run_ids: List[Slug]


class SweepReport(StrictModel):
    sweep_id: Slug
    plan_sha256: Sha256
    profile_id: Slug
    aggregates: List[SweepAggregate]
    result_hashes: Dict[Slug, Sha256]
    warnings: List[str]
