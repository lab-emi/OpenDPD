"""Measurement sessions reference existing captures, never copy or redefine their metrics."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .common import MetricValue, Sha256, Slug, StrictModel, utcnow
from .rf import RFConditions


class InstrumentRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)
    instrument_id: Slug
    role: Literal["generator", "receiver", "attenuator", "supply", "other"]
    model: str = Field(min_length=1, max_length=200)
    serial: Optional[str] = Field(default=None, max_length=100)
    gain_db: Optional[float] = None
    bandwidth_hz: Optional[float] = Field(default=None, gt=0)
    noise_floor_dbm: Optional[float] = None
    compression_input_dbm: Optional[float] = None
    notes: Optional[str] = Field(default=None, max_length=1000)


class CalibrationRecord(StrictModel):
    calibration_id: Slug
    version: str = Field(min_length=1, max_length=100)
    performed_at: AwareDatetime
    method: str = Field(min_length=1, max_length=2000)
    reference_plane: str = Field(min_length=1, max_length=500)
    fixture: Optional[str] = Field(default=None, max_length=500)
    deembedding: Optional[str] = Field(default=None, max_length=1000)
    evidence_sha256: Optional[Sha256] = None


class SessionCapture(StrictModel):
    capture_id: Slug
    # An acquisition may yield several files or run evaluations. It counts once.
    acquisition_id: Slug
    run_id: Slug
    role: Literal["with_dpd", "without_dpd"]
    acquired_at: AwareDatetime
    calibration_id: Optional[Slug] = None
    raw_units: Optional[str] = Field(default=None, max_length=100)
    excluded_reason: Optional[str] = Field(default=None, min_length=1, max_length=1000)


class MeasurementSessionSpec(StrictModel):
    version: Literal["measurement-session-v1"] = "measurement-session-v1"
    title: str = Field(min_length=1, max_length=200)
    dut: str = Field(min_length=1, max_length=200)
    source: Literal["measured", "mock"] = "measured"
    profile_id: Slug
    reference_plane: Optional[str] = Field(default=None, max_length=500)
    operator: Optional[str] = Field(default=None, max_length=100)
    conditions: Optional[RFConditions] = None
    instruments: List[InstrumentRecord] = Field(default_factory=list, max_length=64)
    calibrations: List[CalibrationRecord] = Field(default_factory=list, max_length=64)
    captures: List[SessionCapture] = Field(min_length=1, max_length=256)
    # Explicit predeclared protocol; absent means matching is not established.
    power_tolerance_db: Optional[float] = Field(default=None, ge=0, le=10, allow_inf_nan=False)
    interval: Literal["none", "student_t_95"] = "none"
    uncertainty_budget: Optional[str] = Field(default=None, max_length=4000)
    notes: Optional[str] = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _identities(self):
        for items, key in ((self.captures, "capture_id"), (self.calibrations, "calibration_id"), (self.instruments, "instrument_id")):
            values = [getattr(item, key) for item in items]
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {key}")
        refs = [(c.run_id, c.role) for c in self.captures]
        if len(set(refs)) != len(refs):
            raise ValueError("the same run/capture may be included only once")
        calibration_ids = {c.calibration_id for c in self.calibrations}
        if any(c.calibration_id is not None and c.calibration_id not in calibration_ids for c in self.captures):
            raise ValueError("capture references an unknown calibration")
        return self


class CaptureReview(StrictModel):
    capture: SessionCapture
    status: Literal["included", "excluded", "failed"]
    reason: Optional[str] = None
    raw_sha256: Optional[Sha256] = None
    played_sha256: Optional[Sha256] = None
    declared_output_power_dbm: Optional[float] = None
    processing: Dict[str, object] = Field(default_factory=dict)
    metrics: List[MetricValue] = Field(default_factory=list)
    seed: Optional[int] = None


class RepeatStatistic(StrictModel):
    role: Literal["with_dpd", "without_dpd"]
    metric: str
    unit: str
    n_independent_captures: int
    n_seeds: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    sample_std: Optional[float] = None
    first_to_last_drift: Optional[float] = None
    ci95: Optional[List[float]] = None
    method: str


class MeasurementSession(StrictModel):
    session_id: Slug
    created_at: datetime = Field(default_factory=utcnow)
    spec: MeasurementSessionSpec
    # Bind original records; later metadata edits cannot silently change a session.
    result_hashes: Dict[Slug, Sha256]
    captures: List[CaptureReview]
    repeats: List[RepeatStatistic]
    power_matching: str
    warnings: List[str] = Field(default_factory=list)
