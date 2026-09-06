"""Multi-condition adaptation protocol ``conditions-v1`` (plan S17).

A *condition set* is the data card: one device, one varied dimension, several
conditions that each come from their own capture, with the roles (source /
target) declared before any run. An *adaptation plan* pre-registers which
models are trained on the source and how they meet every target: with no
update at all (``zero_update``), with a bounded amount of new data
(``few_shot``, one run per budget) or trained from scratch on the target
(``full_retrain``). The report keeps every cell, including the failed ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .benchmark import MetricStats
from .benchmark import canonical_sha256
from .common import SCHEMA_VERSION, Sha256, Slug, SoftwareProvenance, StrictModel, utcnow
from .experiment import ModelSpec, ParamValue, TaskType, TrainingConfig

PROTOCOL_ID = "conditions-v1"
MIN_CONDITIONS_FOR_EVIDENCE = 3      # plan S17: at least three conditions along one real dimension, from independent captures
HELD_OUT_POLICY = ("roles are declared in the card before any run; target conditions are never used to select "
                   "hyper-parameters, and every number is read from the target's test split after the run finished")

AdaptationTask = Literal["zero_update", "few_shot", "full_retrain"]
ALL_TASKS: List[AdaptationTask] = ["zero_update", "few_shot", "full_retrain"]


class Condition(StrictModel):
    """One operating condition = one registered dataset from its own capture."""

    condition_id: Slug
    dataset_id: Slug
    role: Literal["source", "target"]
    capture_batch: str = Field(min_length=1, max_length=100)     # the independent acquisition this dataset comes from
    values: Dict[str, ParamValue] = Field(default_factory=dict)  # the varied dimension's value(s), e.g. {"output_power_dbm": 30}
    notes: Optional[str] = Field(default=None, max_length=1000)


class ConditionSet(StrictModel):
    """The data card. ``card_sha256`` covers everything but ``created_at`` and itself."""

    schema_version: int = SCHEMA_VERSION
    set_id: Slug
    device: str = Field(min_length=1, max_length=200)             # the one PA every condition comes from
    dimension: str = Field(min_length=1, max_length=100)          # what varies: capture_batch, output_power_dbm, ...
    conditions: List[Condition] = Field(min_length=2)
    held_out_policy: str = HELD_OUT_POLICY
    created_at: datetime = Field(default_factory=utcnow)
    card_sha256: Optional[Sha256] = None

    @model_validator(mode="after")
    def _well_formed(self) -> "ConditionSet":
        ids = [c.condition_id for c in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError("condition ids must be unique")
        datasets = [c.dataset_id for c in self.conditions]
        if len(set(datasets)) != len(datasets):
            raise ValueError("every condition needs its own dataset: one capture cannot be two conditions")
        if sum(c.role == "source" for c in self.conditions) != 1:
            raise ValueError("exactly one condition is the source")
        return self

    @property
    def source(self) -> Condition:
        return next(c for c in self.conditions if c.role == "source")

    @property
    def targets(self) -> List[Condition]:
        return [c for c in self.conditions if c.role == "target"]

    @property
    def independent_batches(self) -> bool:
        batches = [c.capture_batch for c in self.conditions]
        return len(set(batches)) == len(batches)

    def registered_content(self) -> Dict[str, object]:
        return self.model_dump(mode="json", exclude={"created_at", "card_sha256"})

    def compute_sha256(self) -> str:
        return canonical_sha256(self.registered_content())


class AdaptationEntry(StrictModel):
    """A method: a PA model, or a DPD model that goes through the PA entry's model of the same condition and seed."""

    entry_id: Slug
    task: Literal[TaskType.train_pa, TaskType.train_dpd]
    recipe_id: Optional[Slug] = None
    model: ModelSpec
    training: TrainingConfig
    pa_entry: Optional[Slug] = None

    @model_validator(mode="after")
    def _roles(self) -> "AdaptationEntry":
        if self.task == TaskType.train_pa and self.pa_entry is not None:
            raise ValueError("a PA entry has no pa_entry")
        if self.task == TaskType.train_dpd and self.pa_entry is None:
            raise ValueError("a DPD entry needs pa_entry (the PA entry whose model of the same condition is its surrogate)")
        if self.training.train_samples is not None:
            raise ValueError("the plan's budgets set training.train_samples per cell; the entry leaves it unset")
        return self


class TargetRule(StrictModel):
    """What 'reaching the target' means: one metric of the plan's profile against a threshold."""

    metric: str = Field(min_length=1)
    threshold: float
    better: Literal["lower", "higher"] = "lower"

    def reached(self, value: Optional[float]) -> Optional[bool]:
        if value is None:
            return None
        return value <= self.threshold if self.better == "lower" else value >= self.threshold


class AdaptationPlan(StrictModel):
    """Pre-registered: ``plan_sha256`` covers everything but ``created_at`` and itself."""

    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["conditions-v1"] = PROTOCOL_ID
    condition_set: ConditionSet
    entries: List[AdaptationEntry] = Field(min_length=1)
    tasks: List[AdaptationTask] = Field(default_factory=lambda: list(ALL_TASKS), min_length=1)
    budgets: List[int] = Field(default_factory=lambda: [2000], min_length=1)   # few_shot: samples of the target's train split
    seeds: List[int] = Field(default_factory=lambda: [0], min_length=1)
    metric_profile_id: Slug = "legacy-opendpd-v1"
    device: str = "cpu"
    target: Optional[TargetRule] = None
    held_out_policy: str = HELD_OUT_POLICY
    created_at: datetime = Field(default_factory=utcnow)
    plan_sha256: Optional[Sha256] = None

    @model_validator(mode="after")
    def _consistent(self) -> "AdaptationPlan":
        if self.condition_set.card_sha256 is None:
            raise ValueError("the condition set must be sealed (card_sha256) before a plan is registered")
        ids = [e.entry_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("entry ids must be unique")
        pa_ids = {e.entry_id for e in self.entries if e.task == TaskType.train_pa}
        for e in self.entries:
            if e.pa_entry is not None and e.pa_entry not in pa_ids:
                raise ValueError(f"entry '{e.entry_id}' refers to PA entry '{e.pa_entry}', which is not in the plan")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be distinct")
        if len(set(self.budgets)) != len(self.budgets) or any(b < 1 for b in self.budgets):
            raise ValueError("budgets must be distinct positive sample counts")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self

    def registered_content(self) -> Dict[str, object]:
        return self.model_dump(mode="json", exclude={"created_at", "plan_sha256"})

    def compute_sha256(self) -> str:
        return canonical_sha256(self.registered_content())


class ConditionAudit(StrictModel):
    condition_id: Slug
    dataset_id: Slug
    role: Literal["source", "target"]
    capture_batch: str
    values: Dict[str, ParamValue] = Field(default_factory=dict)
    origin: str                                                  # measured / synthetic / unknown, from the manifest
    raw_sha256: Optional[Sha256] = None
    n_samples: Optional[int] = None
    train_samples: Optional[int] = None                          # size of the train split (the full-retrain data amount)


class AdaptationCell(StrictModel):
    """One (entry, task, condition, budget, seed): a run and what it produced, or why there is no number."""

    entry_id: Slug
    task: AdaptationTask
    condition_id: Slug
    budget_samples: Optional[int] = None                         # few_shot only
    seed: int
    run_id: Optional[Slug] = None
    status: Literal["ok", "failed", "missing"]
    failure: Optional[str] = None
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    new_samples: int = Field(ge=0)                               # samples of the condition used to obtain the weights
    wall_clock_s: Optional[float] = None
    device: str
    config_sha256: Optional[Sha256] = None
    checkpoint_sha256: Optional[Sha256] = None
    reached_target: Optional[bool] = None


class CellAggregate(StrictModel):
    entry_id: Slug
    task: AdaptationTask
    condition_id: Slug
    budget_samples: Optional[int] = None
    n_seeds: int = Field(ge=1)
    n_ok: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    metrics: Dict[str, MetricStats] = Field(default_factory=dict)
    new_samples: int = Field(ge=0)
    mean_wall_clock_s: Optional[float] = None
    target_reached_fraction: Optional[float] = Field(default=None, ge=0, le=1)


class EvidenceBar(StrictModel):
    """Whether the card meets the S17 evidence bar; a report below the bar is a rehearsal, not evidence."""

    min_conditions: int = MIN_CONDITIONS_FOR_EVIDENCE
    n_conditions: int
    independent_batches: bool
    measured_origin: bool
    met: bool


class AdaptationReport(StrictModel):
    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["conditions-v1"] = PROTOCOL_ID
    plan_sha256: Sha256
    condition_set: ConditionSet
    conditions: List[ConditionAudit]
    metric_profile_id: Slug
    metric_profile_version: int
    device: str
    seeds: List[int]
    budgets: List[int]
    target: Optional[TargetRule] = None
    cells: List[AdaptationCell]
    aggregates: List[CellAggregate]
    evidence_bar: EvidenceBar
    repeats: str                                                 # seeds vs capture batches, stated separately
    limitations: List[str] = Field(default_factory=list)
    software: SoftwareProvenance
    machine: Dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utcnow)
    report_sha256: Optional[Sha256] = None

    def content_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"report_sha256"}))

    def sealed(self) -> "AdaptationReport":
        return self.model_copy(update={"report_sha256": self.content_sha256()})

    @property
    def intact(self) -> bool:
        return self.report_sha256 is not None and self.report_sha256 == self.content_sha256()

    def summary(self) -> "AdaptationReportSummary":
        return AdaptationReportSummary(
            plan_sha256=self.plan_sha256, set_id=self.condition_set.set_id, device=self.condition_set.device,
            dimension=self.condition_set.dimension, n_conditions=len(self.condition_set.conditions),
            n_cells=len(self.cells), n_without_number=sum(c.status != "ok" for c in self.cells),
            evidence_met=self.evidence_bar.met, generated_at=self.generated_at, report_sha256=self.report_sha256)


class AdaptationReportSummary(StrictModel):
    """One row of the reports list: enough to pick a report, never a number to compare."""

    plan_sha256: Sha256
    set_id: Slug
    device: str
    dimension: str
    n_conditions: int
    n_cells: int
    n_without_number: int
    evidence_met: bool
    generated_at: datetime
    report_sha256: Optional[Sha256] = None
