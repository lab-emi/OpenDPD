"""Benchmark protocol ``benchmark-v1`` (plan S12): pre-registered plans, per-seed and aggregate reports whose numbers
are bound to run ids and hashes, and regression baselines that only a human can approve."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .common import SCHEMA_VERSION, Sha256, Slug, SoftwareProvenance, StrictModel, utcnow
from .experiment import DatasetRef, ModelSpec, TaskType, TrainingConfig

PROTOCOL_ID = "benchmark-v1"
MIN_SEEDS = 3
BenchmarkTier = Literal["cpu_regression", "gpu_full"]

SELECTION_RULE = ("the checkpoint is selected on the validation split by the protocol metric of the task "
                  "(PA: NMSE, DPD: ACLR average); the test split is scored once, by the evaluation stage, and never "
                  "used for selection or tuning")
TEST_SET_POLICY = ("every entry and seed is scored on the test split exactly once from its selected checkpoint; "
                   "re-evaluation under another metric profile re-reads the same checkpoint and is a deterministic "
                   "regression, not a new attempt")
SEED_NOTE = ("seeds are pre-registered in the plan; per-seed values are reported next to the aggregate; "
             "the spread of {n} seeds estimates run-to-run variation on this data and hardware and is not evidence "
             "of generality across devices, signals or operating points")
COMPUTE_NOTE = ("equal parameter counts are not equal compute cost: families differ in operations per sample, "
                "memory and look-ahead; a least-squares fit and gradient training are different procedures with "
                "different budgets, and ILA and DLA are different training paths")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


class BenchmarkEntry(StrictModel):
    """One model under a fixed budget; the plan's seeds replace ``training.seed``."""

    entry_id: Slug
    task: TaskType
    recipe_id: Optional[Slug] = None
    model: ModelSpec
    training: TrainingConfig
    pa_entry: Optional[Slug] = None          # DPD entries: the PA entry whose run of the same seed is the surrogate

    @model_validator(mode="after")
    def _roles(self) -> "BenchmarkEntry":
        if self.task == TaskType.train_pa and self.pa_entry is not None:
            raise ValueError("a PA entry has no pa_entry")
        if self.task == TaskType.train_dpd and self.pa_entry is None:
            raise ValueError("a DPD entry needs pa_entry (the PA entry that provides its surrogate)")
        if self.task == TaskType.run_dpd:
            raise ValueError("benchmark entries train (or fit) models; run_dpd is not an entry")
        return self


class BenchmarkPlan(StrictModel):
    """Pre-registered: the hash of everything below except ``created_at`` and ``plan_sha256`` identifies the plan."""

    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["benchmark-v1"] = PROTOCOL_ID
    tier: BenchmarkTier
    dataset: DatasetRef
    metric_profile_id: Slug = "legacy-opendpd-v1"
    device: str = "cpu"
    seeds: List[int] = Field(min_length=MIN_SEEDS)
    entries: List[BenchmarkEntry] = Field(min_length=1)
    selection_rule: str = SELECTION_RULE
    test_set_policy: str = TEST_SET_POLICY
    created_at: datetime = Field(default_factory=utcnow)
    plan_sha256: Optional[Sha256] = None

    @model_validator(mode="after")
    def _consistent(self) -> "BenchmarkPlan":
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        ids = [e.entry_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("entry ids must be unique")
        pa_ids = {e.entry_id for e in self.entries if e.task == TaskType.train_pa}
        for e in self.entries:
            if e.pa_entry is not None and e.pa_entry not in pa_ids:
                raise ValueError(f"entry '{e.entry_id}' refers to PA entry '{e.pa_entry}', which is not in the plan")
        return self

    def registered_content(self) -> Dict[str, object]:
        return self.model_dump(mode="json", exclude={"created_at", "plan_sha256"})

    def compute_sha256(self) -> str:
        return canonical_sha256(self.registered_content())


class MetricStats(StrictModel):
    n: int = Field(ge=1)
    mean: float
    std: Optional[float] = None              # sample standard deviation; None with a single value
    min: float
    max: float


class SeedScore(StrictModel):
    seed: int
    run_id: Slug
    config_sha256: Sha256
    checkpoint_sha256: Optional[Sha256] = None
    surrogate_run_id: Optional[Slug] = None
    selected_epoch: Optional[int] = None
    metrics: Dict[str, Optional[float]]
    wall_clock_s: Optional[float] = None


class EntryResult(StrictModel):
    entry_id: Slug
    task: TaskType
    model: ModelSpec
    training_path: Optional[str] = None
    n_parameters: Optional[int] = None
    lookahead_samples: Optional[int] = None
    execution_semantics: str = "offline_segmented"
    fit: Optional[Dict[str, object]] = None  # least-squares diagnostics (identical across seeds: no seed is used)
    seeds: List[SeedScore] = Field(default_factory=list)
    missing_seeds: List[int] = Field(default_factory=list)
    aggregate: Dict[str, MetricStats] = Field(default_factory=dict)


class DataAudit(StrictModel):
    dataset_id: Slug
    raw_sha256: Optional[Sha256] = None
    preprocessing_version: str
    split_version: str
    guard_samples: Optional[int] = None
    surrogate_training_split: str = "train"
    dpd_optimisation: str = ("train split: through the frozen PA surrogate of the same seed (gradient, DLA) or on the "
                             "measured data (least squares, ILA)")
    selection_split: str = "val"
    reported_split: str = "test"
    statement: str = ("the surrogate is a model fitted to measured data; agreement between a DPD's simulated and "
                      "measured outcome is not established by this benchmark (cross-validation of the surrogate is "
                      "not physical validation)")


class BenchmarkReport(StrictModel):
    """Machine-readable, hash-bound: every number is copied from a run's stored result; ``report_sha256`` covers
    everything else so an edited number is detectable."""

    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["benchmark-v1"] = PROTOCOL_ID
    tier: BenchmarkTier
    plan_sha256: Sha256
    dataset: DataAudit
    metric_profile_id: Slug
    metric_profile_version: int
    device: str
    seeds: List[int]
    entries: List[EntryResult]
    software: SoftwareProvenance
    machine: Dict[str, str] = Field(default_factory=dict)      # cpu / cores / os — never a hostname or a path
    generated_at: datetime = Field(default_factory=utcnow)
    notes: List[str] = Field(default_factory=list)
    report_sha256: Optional[Sha256] = None

    def content_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"report_sha256"}))

    def sealed(self) -> "BenchmarkReport":
        return self.model_copy(update={"report_sha256": self.content_sha256()})

    @property
    def intact(self) -> bool:
        return self.report_sha256 is not None and self.report_sha256 == self.content_sha256()


class BaselineBand(StrictModel):
    reference: float
    tolerance: float = Field(ge=0)          # absolute, in the metric's unit (dB)
    worse_is: Literal["higher", "lower"]


class RegressionBaseline(StrictModel):
    """Approved numbers a report is checked against. Unapproved baselines are advisory: they never block."""

    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["benchmark-v1"] = PROTOCOL_ID
    tier: BenchmarkTier
    plan_sha256: Sha256
    entries: Dict[Slug, Dict[str, BaselineBand]]
    basis: str = Field(min_length=1)        # how reference and tolerance were derived (report, machine, spread)
    approved_by: Optional[str] = None
    approved_on: Optional[date] = None
    notes: List[str] = Field(default_factory=list)

    @property
    def approved(self) -> bool:
        return bool(self.approved_by) and self.approved_on is not None


RegressionStatus = Literal["within", "degraded", "improved", "missing"]


class RegressionItem(StrictModel):
    entry_id: Slug
    metric: str
    reference: float
    tolerance: float
    worse_is: Literal["higher", "lower"]
    observed: Optional[float] = None
    delta: Optional[float] = None
    status: RegressionStatus


class RegressionCheck(StrictModel):
    ok: bool                                # no metric degraded beyond its band and nothing missing
    approved: bool                          # the baseline carries a human approval
    blocking: bool                          # not ok and approved: release must stop
    items: List[RegressionItem]
    verdict: str
