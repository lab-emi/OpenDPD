"""Cost evidence is qualified by implementation, workload, precision and source."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .common import FileRef, MetricValue, Sha256, Slug, StrictModel, utcnow

CostSource = Literal['operation_count', 'cpu_reference_timing', 'fpga_synthesis', 'fpga_board_measurement',
                     'asic_synthesis', 'post_layout_simulation', 'chip_measurement']


class CostValues(StrictModel):
    # Storage is model constants only; state/buffers and runtime allocation are separate.
    constant_bytes: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    state_bytes: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    input_buffer_bytes: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    mac_per_sample: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    dynamic_skip_fraction: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    table_bytes: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    table_lookups_per_sample: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    throughput_samples_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    latency_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    power_w: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    energy_j_per_sample: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    area_mm2: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    lut_count: int | None = Field(default=None, ge=0)


class PrecisionModule(StrictModel):
    module: str = Field(min_length=1, max_length=120)
    format: str = Field(min_length=1, max_length=200)


class HardwareCostDraft(StrictModel):
    run_id: Slug
    profile_id: Slug
    title: str = Field(min_length=1, max_length=160)
    source_kind: CostSource
    report_sha256: Sha256
    target: str = Field(min_length=1, max_length=200)
    process: str | None = Field(default=None, max_length=160)
    clock_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    batch_size: int = Field(ge=1)
    activity: str = Field(min_length=1, max_length=400)
    boundary: str = Field(min_length=1, max_length=800)
    precision: list[PrecisionModule] = Field(min_length=1, max_length=30)
    values: CostValues
    synthetic: bool = False
    notes: str = Field(default='', max_length=2000)

    @model_validator(mode='after')
    def _basis(self):
        if not any(value is not None for value in self.values.model_dump().values()):
            raise ValueError('enter at least one sourced cost value')
        if self.source_kind in {'operation_count', 'cpu_reference_timing'} and (self.values.power_w is not None or self.values.energy_j_per_sample is not None):
            raise ValueError('operation counts and CPU timing alone cannot establish hardware power or energy')
        if self.source_kind in {'asic_synthesis', 'post_layout_simulation', 'chip_measurement'} and not self.process:
            raise ValueError('ASIC cost evidence must name its process, or explicitly state that it was not disclosed')
        return self


class CostAttachment(StrictModel):
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    filename: str


class HardwareCostEntry(StrictModel):
    entry_id: str
    run_id: Slug
    profile_id: Slug
    title: str
    source_kind: CostSource
    source_file: FileRef
    source_type: Literal['checkpoint_shapes', 'studio_fixed_point_report', 'user_report']
    result_sha256: Sha256
    weights_sha256: Sha256
    values: CostValues
    precision: list[PrecisionModule]
    target: str
    process: str | None = None
    clock_hz: float | None = None
    batch_size: int | None = None
    activity: str
    boundary: str
    metrics: list[MetricValue]
    metric_basis: str
    rf_evidence_type: str
    execution_semantics: str
    synthetic: bool = False
    stored_parameter_count: int | None = None
    stored_tensor_elements: int | None = None
    lookahead_samples: int | None = None
    lookahead_lower_bound_s: float | None = None
    warmup_samples: int | None = None
    limitations: list[str]
    created_at: datetime = Field(default_factory=utcnow)
    stale: bool = False


class HardwareCostReport(StrictModel):
    protocol_id: Literal['cost-ledger-v1'] = 'cost-ledger-v1'
    entries: list[HardwareCostEntry]
    missing: dict[str, str]
    notes: list[str]
    comparison_notes: list[str] = Field(default_factory=list)
