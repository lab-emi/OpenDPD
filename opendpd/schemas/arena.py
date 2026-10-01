"""Executable Arena contracts: clients request evaluation, never supply scores."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .common import Sha256, Slug, StrictModel, utcnow

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
ArenaStatus = Literal["queued", "running", "succeeded", "failed", "interrupted"]


class ArenaSubmissionRequest(StrictModel):
    board_id: Slug
    backbone: Slug
    backbone_id: str | None = Field(default=None, pattern=r"^ub-[a-f0-9]{64}$")
    display_name: str = Field(min_length=1, max_length=80)
    accepted_protocol_sha256: Sha256

    @field_validator("display_name")
    @classmethod
    def _name(cls, value):
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("display_name must be nonempty text without control characters")
        return value

    @model_validator(mode="after")
    def _custom_reference(self):
        if self.backbone != "user_template" and self.backbone_id is not None:
            raise ValueError("only user_template accepts backbone_id")
        return self


class ArenaBoard(StrictModel):
    board_id: Slug
    title: str
    description: str
    evidence_type: Literal["synthetic_simulation", "measured_data_simulation"]
    evidence_label: str
    dataset: str
    conditions: list[str]


class ArenaBackbone(StrictModel):
    key: Slug
    display_name: str
    family: str
    deterministic: bool = False


class ArenaRule(StrictModel):
    title: str
    description: str


class ArenaRanking(StrictModel):
    ranking_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    title: str
    description: str
    unit: str


class ArenaPA(StrictModel):
    model: str
    hidden_size: int
    parameters: int
    validation_nmse_db: FiniteFloat
    test_nmse_db: FiniteFloat
    checkpoint_sha256: Sha256


class ArenaProtocol(StrictModel):
    protocol_id: str
    protocol_sha256: Sha256
    training_sha256: Sha256
    title: str
    description: str
    boards: list[ArenaBoard] = Field(min_length=1, max_length=4)
    seeds: list[int] = Field(min_length=1)
    budgets: list[int] = Field(min_length=1, max_length=8)
    rules: list[ArenaRule]
    rankings: list[ArenaRanking] = Field(min_length=1)
    score_formula: str
    training: dict[str, JsonValue]
    scoring: dict[str, JsonValue]
    cost_model: dict[str, JsonValue]
    pa_models: dict[str, ArenaPA] = Field(default_factory=dict)


class ArenaMetricSummary(StrictModel):
    """Means over conditions and seeds of one configuration under one frozen PA.
    Symbol EVM and output ACLR determine quality; AER and NMSE are diagnostics.
    EVM percent is converted from mean dB (a geometric mean), not pooled across seeds.
    """
    nmse_db: FiniteFloat
    aclr_db: FiniteFloat
    evm_db: FiniteFloat | None = None
    evm_pct: FiniteFloat | None = None
    baseline_evm_db: FiniteFloat | None = None
    baseline_evm_pct: FiniteFloat | None = None
    evm_improvement_db: FiniteFloat | None = None
    ib_error_db: FiniteFloat | None = None
    baseline_nmse_db: FiniteFloat
    baseline_aclr_db: FiniteFloat
    nmse_improvement_db: FiniteFloat
    aclr_improvement_db: FiniteFloat
    aer_db: FiniteFloat | None = None
    baseline_aer_db: FiniteFloat | None = None
    aer_improvement_db: FiniteFloat | None = None


class ArenaBudgetResult(StrictModel):
    """One point of the parameter sweep: its configuration, analytic cost and gated scores."""
    budget: int = Field(ge=1)
    available: bool = True
    model_parameters: dict[str, JsonValue] | None = None
    parameters: int | None = Field(default=None, ge=0)
    mul: int | None = Field(default=None, ge=0)
    add: int | None = Field(default=None, ge=0)
    ops: int | None = Field(default=None, ge=0)
    nonlinear: dict[str, int] = Field(default_factory=dict)
    nonlinear_mul: int | None = Field(default=None, ge=0)
    nonlinear_add: int | None = Field(default=None, ge=0)
    operation_items: list[dict[str, JsonValue]] = Field(default_factory=list)
    parameter_ratio: FiniteFloat | None = Field(default=None, gt=0)
    operation_ratio: FiniteFloat | None = Field(default=None, gt=0)
    qualified: bool = False
    reasons: list[str] = Field(default_factory=list)
    quality_db: FiniteFloat | None = None
    quality_std_db: FiniteFloat | None = Field(default=None, ge=0)
    quality_conservative_db: FiniteFloat | None = None
    parameter_efficiency_db: FiniteFloat | None = None
    arithmetic_efficiency_db: FiniteFloat | None = None
    score: FiniteFloat | None = None
    metrics: ArenaMetricSummary | None = None
    expected_cases: int = Field(default=0, ge=0)
    completed_cases: int = Field(default=0, ge=0)


class ArenaRankEntry(StrictModel):
    score: FiniteFloat | None = None
    rank: int | None = Field(default=None, ge=1)


class ArenaRow(StrictModel):
    entry_id: str
    board_id: Slug
    backbone: Slug
    display_name: str
    origin: Literal["official", "workspace"]
    status: ArenaStatus
    protocol_sha256: Sha256
    rank: int | None = Field(default=None, ge=1)
    eligible: bool = False
    eligibility_reasons: list[str] = Field(default_factory=list)
    score: FiniteFloat | None = None
    rankings: dict[str, ArenaRankEntry] = Field(default_factory=dict)
    budgets: list[ArenaBudgetResult] = Field(default_factory=list)
    qualified_budgets: int = Field(default=0, ge=0)
    available_budgets: int = Field(default=0, ge=0)
    best_budget: int | None = Field(default=None, ge=1)
    quality_db: FiniteFloat | None = None
    quality_conservative_db: FiniteFloat | None = None
    metrics: ArenaMetricSummary | None = None
    parameters: int | None = Field(default=None, ge=0)
    ops_per_parameter: FiniteFloat | None = Field(default=None, ge=0)
    execution_semantics: str = "offline_segmented"
    seeds: list[int] = Field(default_factory=list)
    expected_cases: int = Field(default=0, ge=0)
    completed_cases: int = Field(default=0, ge=0)
    evidence_type: Literal["synthetic_simulation", "measured_data_simulation"]
    cases: list[dict[str, JsonValue]] = Field(default_factory=list)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime | None = None


class ArenaCoverage(StrictModel):
    expected: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    missing: list[str] = Field(default_factory=list)


class ArenaLeaderboard(StrictModel):
    board: ArenaBoard
    protocol_sha256: Sha256
    rows: list[ArenaRow]
    coverage: ArenaCoverage
    scope_note: str


class ArenaCatalog(StrictModel):
    protocol: ArenaProtocol
    backbones: list[ArenaBackbone]
    submissions_available: bool = True
    submission_unavailable_reason: str | None = None
    scope_note: str


class ArenaSubmission(StrictModel):
    submission_id: str = Field(pattern=r"^arena-[a-f0-9]{32}$")
    request: ArenaSubmissionRequest
    protocol_sha256: Sha256
    status: ArenaStatus = "queued"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    result: ArenaRow | None = None
    error: str | None = None
    progress: "ArenaProgress | None" = None


class ArenaProgress(StrictModel):
    phase: str = Field(min_length=1, max_length=80)
    epoch: int = Field(default=0, ge=0, le=100000)
    epochs: int = Field(default=0, ge=0, le=100000)
    completed_cases: int = Field(default=0, ge=0, le=10000)
    expected_cases: int = Field(default=0, ge=0, le=10000)
    message: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _bounded_counts(self):
        if self.epoch > self.epochs or self.completed_cases > self.expected_cases:
            raise ValueError("Arena progress exceeds the declared budget")
        return self


ArenaSubmission.model_rebuild()
