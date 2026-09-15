"""Bounded signal inspection, independent of paired training datasets."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from .common import StrictModel, Slug


class AnalyzerSource(StrictModel):
    kind: Literal["generated", "upload", "dataset", "virtual_pa"]
    source_id: Slug
    role: Literal["input", "output"] = "input"
    version: Slug = "raw-v1"

    @model_validator(mode="after")
    def valid_role(self):
        if self.kind != "dataset":
            expected = "output" if self.kind == "virtual_pa" else "input"
            if self.role != expected:
                raise ValueError(f"{self.kind} signals require the {expected} role.")
        return self


class AnalyzerSourceInfo(StrictModel):
    source: AnalyzerSource
    label: str
    sample_count: int
    sample_rate_hz: float | None = None
    bandwidth_hz: float | None = None
    columns: list[str] = Field(default_factory=lambda: ["I", "Q"])
    complex_columns: list[int] = Field(default_factory=list)
    origin: str


class AnalyzerConfig(StrictModel):
    sample_rate_hz: float = Field(default=80e6, ge=1, le=2e9, allow_inf_nan=False)
    bandwidth_hz: float = Field(default=20e6, gt=0, le=2e9, allow_inf_nan=False)
    center_hz: float = Field(default=0, allow_inf_nan=False)
    start_sample: int = Field(default=0, ge=0, le=1_000_000_000)
    n_samples: int = Field(default=262144, ge=256, le=1_000_000)
    fft_size: int = Field(default=4096, ge=256, le=16384)
    window: Literal["hann", "hamming", "blackman", "boxcar"] = "hann"
    overlap: Literal[0, .5, .75] = .5
    occupied_percent: float = Field(default=99, ge=50, le=99.99, allow_inf_nan=False)
    adjacent_offset_hz: float | None = Field(default=None, gt=0, le=2e9, allow_inf_nan=False)
    remove_dc: bool = False
    frequency_shift_hz: float = Field(default=0, allow_inf_nan=False)
    sample_format: Literal["auto", "real", "complex", "iq"] = "auto"
    i_column: int = Field(default=0, ge=0, le=7)
    q_column: int = Field(default=1, ge=0, le=7)
    samples_per_symbol: int = Field(default=8, ge=2, le=128)
    symbol_offset: int = Field(default=0, ge=0, le=127)
    reference_gain_fit: bool = False

    @model_validator(mode="after")
    def valid(self):
        if self.fft_size & (self.fft_size - 1):
            raise ValueError("FFT size must be a power of two.")
        if abs(self.center_hz) + self.bandwidth_hz / 2 > self.sample_rate_hz / 2:
            raise ValueError("The measurement band must fit inside Nyquist.")
        if abs(self.frequency_shift_hz) >= self.sample_rate_hz / 2:
            raise ValueError("Frequency shift must be smaller than half the sample rate.")
        if self.adjacent_offset_hz is not None and self.adjacent_offset_hz < self.bandwidth_hz:
            raise ValueError("Adjacent bands must not overlap the main measurement band.")
        if self.symbol_offset >= self.samples_per_symbol:
            raise ValueError("Symbol offset must be smaller than samples per symbol.")
        if self.sample_format == "iq" and self.i_column == self.q_column:
            raise ValueError("I and Q must use different columns.")
        return self


class AnalyzerRequest(StrictModel):
    source: AnalyzerSource
    config: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    reference: AnalyzerSource | None = None


class SignalMeasurement(StrictModel):
    key: str
    label: str
    value: float | None
    unit: str = ""
    reason: str | None = None


class SignalAnalysis(StrictModel):
    schema_version: Literal["signal-analysis-v1"] = "signal-analysis-v1"
    source: AnalyzerSourceInfo
    config: AnalyzerConfig
    source_sha256: str
    reference_sha256: str | None = None
    sample_range: tuple[int, int]
    sample_count: int
    real_signal: bool
    measurements: list[SignalMeasurement]
    frequency_hz: list[float]
    psd_dbfs_hz: list[float]
    time_s: list[float]
    time_i: list[float]
    time_q: list[float]
    envelope: list[float]
    instantaneous_frequency_hz: list[float | None]
    scatter_i: list[float]
    scatter_q: list[float]
    ccdf_db: list[float]
    ccdf_probability: list[float]
    histogram_amplitude: list[float]
    histogram_probability: list[float]
    spectrogram_time_s: list[float]
    spectrogram_frequency_hz: list[float]
    spectrogram_dbfs_hz: list[list[float]]
    eye_i: list[list[float]]
    eye_q: list[list[float]]
    reference_error_percent: list[float] = Field(default_factory=list)
    notes: list[str]
