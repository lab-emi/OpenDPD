"""Versioned, bounded complex-baseband signal generator contracts."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .common import Sha256, Slug, StrictModel
from .dataset import DatasetManifest


class GeneratorConfig(StrictModel):
    version: Literal["signal-generator-v1"] = "signal-generator-v1"
    preset_id: str = Field(default="custom-ofdm", max_length=64)
    waveform: Literal["ofdm", "qam", "psk", "fsk", "gfsk", "tone", "multitone", "chirp", "noise"] = "ofdm"
    sample_rate_hz: float = Field(default=80e6, ge=1000, le=2e9, allow_inf_nan=False)
    bandwidth_hz: float = Field(default=20e6, gt=0, le=1e9, allow_inf_nan=False)
    carrier_frequency_hz: float = Field(default=3.5e9, ge=0, le=110e9, allow_inf_nan=False)
    length_mode: Literal["samples", "duration"] = "samples"
    n_samples: int = Field(default=131072, ge=256, le=1_000_000)
    duration_ms: float = Field(default=1, gt=0, le=1000, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    rms: float = Field(default=.2, ge=.001, le=1, allow_inf_nan=False)
    fft_size: int = Field(default=256, ge=64, le=16384)
    oversampling: Literal[1, 2, 4, 8] = 4
    channel_subcarriers: list[int] = Field(default_factory=lambda: [242], min_length=1, max_length=16)
    channel_modulations: list[int] = Field(default_factory=lambda: [64], min_length=1, max_length=16)
    channel_power_db: list[float] = Field(default_factory=lambda: [0.], min_length=1, max_length=16)
    shared_channel_settings: bool | None = None
    channel_gap_bins: int = Field(default=0, ge=0, le=1024)
    dc_null: bool = True
    pilot_mode: Literal["comb", "explicit", "none"] = "comb"
    pilot_spacing: int = Field(default=16, ge=2, le=1024)
    pilot_indices: list[int] = Field(default_factory=list, max_length=8192)
    pilot_boost_db: float = Field(default=0, ge=-20, le=20, allow_inf_nan=False)
    cp_mode: Literal["fixed", "nr_normal", "nr_extended"] = "fixed"
    cp_samples: int = Field(default=16, ge=0, le=16384)
    modulation_order: Literal[2, 4, 16, 64, 256, 1024, 4096] = 64
    samples_per_symbol: int = Field(default=8, ge=2, le=64)
    rrc_rolloff: float = Field(default=.25, ge=0, le=1, allow_inf_nan=False)
    rrc_span_symbols: int = Field(default=10, ge=4, le=32)
    psk_order: Literal[2, 4, 8, 16, 32] = 8
    payload_mode: Literal["random", "prbs9", "prbs15", "bits"] = "random"
    payload_bits: str = Field(default="00110101", min_length=1, max_length=4096, pattern=r"^[01]+$")
    dft_spreading: bool = False
    fsk_deviation_hz: float = Field(default=1e6, gt=0, le=1e9, allow_inf_nan=False)
    gaussian_bt: float = Field(default=.5, ge=.1, le=2, allow_inf_nan=False)
    tone_frequency_hz: float = Field(default=1e6, allow_inf_nan=False)
    tone_count: int = Field(default=8, ge=2, le=64)
    multitone_phase: Literal["random", "coherent", "schroeder"] = "random"
    phase_offset_deg: float = Field(default=0, ge=-360, le=360, allow_inf_nan=False)
    phase_noise_rms_deg: float = Field(default=0, ge=0, le=30, allow_inf_nan=False)
    burst_on_samples: int | None = Field(default=None, ge=16, le=1_000_000)
    burst_off_samples: int = Field(default=1024, ge=0, le=1_000_000)
    burst_ramp_samples: int = Field(default=32, ge=0, le=65536)
    frequency_offset_hz: float = Field(default=0, allow_inf_nan=False)
    iq_gain_db: float = Field(default=0, ge=-6, le=6, allow_inf_nan=False)
    iq_phase_deg: float = Field(default=0, ge=-30, le=30, allow_inf_nan=False)
    dc_i: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False)
    dc_q: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False)
    snr_db: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    clip_db: float | None = Field(default=None, ge=0, le=30, allow_inf_nan=False)

    @property
    def sample_count(self) -> int:
        return self.n_samples if self.length_mode == "samples" else math.floor(self.sample_rate_hz * self.duration_ms / 1000 + .5)

    @model_validator(mode="after")
    def _valid(self):
        if not 256 <= self.sample_count <= 1_000_000:
            raise ValueError("Choose a duration producing 256–1,000,000 complex I/Q samples.")
        if self.bandwidth_hz > self.sample_rate_hz:
            raise ValueError("Baseband bandwidth must not exceed the sample rate.")
        if self.fft_size & (self.fft_size - 1):
            raise ValueError("FFT size must be a power of two.")
        count = len(self.channel_subcarriers)
        if len(self.channel_modulations) != count or len(self.channel_power_db) != count:
            raise ValueError("Each OFDMA channel needs its own subcarrier count, modulation and power.")
        if any(n < 2 or n > 16382 for n in self.channel_subcarriers):
            raise ValueError("Each channel must contain 2–16,382 subcarriers including pilots.")
        if any(m not in (2, 4, 16, 64, 256, 1024, 4096) for m in self.channel_modulations):
            raise ValueError("Use BPSK or square QAM orders 4, 16, 64, 256, 1024 or 4096.")
        if any(not math.isfinite(p) or abs(p) > 40 for p in self.channel_power_db):
            raise ValueError("Channel powers must be finite and within ±40 dB.")
        same = all(len(set(values)) == 1 for values in (self.channel_subcarriers, self.channel_modulations, self.channel_power_db))
        if self.shared_channel_settings is None:
            self.shared_channel_settings = same
        elif self.shared_channel_settings and not same:
            raise ValueError("Shared OFDMA settings require identical subcarrier counts, modulation and power across channels.")
        if len(set(self.pilot_indices)) != len(self.pilot_indices):
            raise ValueError("Pilot carrier indices must be unique.")
        if self.waveform == "ofdm":
            if self.dft_spreading and self.pilot_mode != "none":
                raise ValueError("DFT spreading requires pilot mode None; transform-spread reference pilots are not implemented.")
            span = sum(self.channel_subcarriers) + (count - 1) * self.channel_gap_bins
            occupied = span + int(self.dc_null)
            spacing = self.sample_rate_hz / (self.fft_size * self.oversampling)
            if occupied > self.fft_size - 1:
                raise ValueError("OFDMA channels, gaps and DC null do not fit the FFT. Increase FFT size or reduce carriers.")
            outer_bin = max(span // 2, math.ceil(span / 2) + int(self.dc_null) - 1)
            allocated_extent = (outer_bin + .5) * spacing
            if 2 * allocated_extent > self.bandwidth_hz * (1 + 1e-12):
                raise ValueError("Allocated subcarriers exceed the declared baseband bandwidth.")
            if self.cp_samples > self.fft_size:
                raise ValueError("Cyclic prefix must not exceed the FFT size.")
            if self.cp_mode.startswith("nr_"):
                mu = math.log2(spacing / 15000)
                if abs(mu - round(mu)) > 1e-9 or round(mu) not in range(5):
                    raise ValueError("NR cyclic prefix requires 15, 30, 60, 120 or 240 kHz subcarrier spacing.")
                if self.cp_mode == "nr_extended" and round(mu) != 2:
                    raise ValueError("NR extended cyclic prefix requires 60 kHz spacing.")
                if self.fft_size < 128:
                    raise ValueError("NR cyclic prefix needs at least 128 FFT bins.")
        if self.waveform in ("qam", "psk") and self.rrc_span_symbols * self.samples_per_symbol % 2:
            raise ValueError("RRC span × samples per symbol must be even.")
        if self.burst_on_samples is not None and 2 * self.burst_ramp_samples > self.burst_on_samples:
            raise ValueError("Burst ramps must fit within the on interval.")
        if self.waveform in ("qam", "psk") and (1 + self.rrc_rolloff) * self.sample_rate_hz / self.samples_per_symbol > self.bandwidth_hz * (1 + 1e-12):
            raise ValueError("RRC QAM bandwidth exceeds the declared bandwidth; increase samples/symbol or bandwidth.")
        if self.waveform in ("fsk", "gfsk") and 2 * (self.fsk_deviation_hz + self.sample_rate_hz / self.samples_per_symbol) > self.bandwidth_hz:
            raise ValueError("The FSK deviation and symbol rate exceed the declared engineering bandwidth estimate.")
        extent = (allocated_extent if self.waveform == "ofdm" else
                  (1 + self.rrc_rolloff) * self.sample_rate_hz / self.samples_per_symbol / 2 if self.waveform in ("qam", "psk") else
                  abs(self.tone_frequency_hz) if self.waveform == "tone" else self.bandwidth_hz / 2)
        if self.waveform == "tone" and extent > self.bandwidth_hz / 2:
            raise ValueError("The tone must lie inside the declared baseband bandwidth.")
        if extent + abs(self.frequency_offset_hz) >= self.sample_rate_hz / 2:
            raise ValueError("The waveform plus frequency offset must lie strictly inside Nyquist.")
        return self


class GeneratorPreset(StrictModel):
    preset_id: str
    family: Literal["nr", "wifi6", "wifi7", "wifi8", "custom"]
    label: str
    description: str
    config: GeneratorConfig


class GeneratorAnalysis(StrictModel):
    sample_count: int
    duration_ms: float
    sample_rate_hz: float
    subcarrier_spacing_hz: float | None
    useful_symbol_us: float | None
    cp_lengths_samples: list[int]
    complete_symbols: int
    trailing_samples: int
    active_carriers: int
    data_carriers: int
    pilot_carriers: int
    rms: float
    peak: float
    papr_db: float
    mean_power_dbfs: float
    occupied_bandwidth_99_hz: float
    dc_magnitude: float
    evm_percent: float | None
    evm_symbols: int
    evm_per_symbol_percent: list[float] = Field(default_factory=list)
    evm_subcarrier_indices: list[int] = Field(default_factory=list)
    evm_per_subcarrier_percent: list[float] = Field(default_factory=list)
    time_us: list[float]
    time_i: list[float]
    time_q: list[float]
    time_envelope: list[float]
    frequency_mhz: list[float]
    psd_dbfs_hz: list[float]
    constellation_i: list[float]
    constellation_q: list[float]
    reference_i: list[float]
    reference_q: list[float]
    ccdf_db: list[float]
    ccdf_probability: list[float]
    allocation: list[dict[str, object]]
    notes: list[str]


class GeneratedSignal(StrictModel):
    kind: Literal["pa_input"] = "pa_input"
    signal_id: str = Field(pattern=r"^sg-[a-f0-9]{64}$")
    config: GeneratorConfig
    iq_sha256: Sha256
    coverage: Literal["numerology", "experimental", "custom"]
    analysis: GeneratorAnalysis
    download_url: str


class GeneratorDatasetRequest(StrictModel):
    dataset_id: Slug
    display_name: str = Field(default="Generated signal · synthetic PA", min_length=1, max_length=180)
    pa_gain: float = Field(default=1.6, ge=.1, le=10, allow_inf_nan=False)
    compression: float = Field(default=.7, ge=0, le=2, allow_inf_nan=False)
    am_pm: float = Field(default=.1, ge=0, le=2, allow_inf_nan=False)
    memory: float = Field(default=.08, ge=0, le=.5, allow_inf_nan=False)
    noise_db: float = Field(default=-60, ge=-120, le=-10, allow_inf_nan=False)
    guard_samples: int = Field(default=256, ge=0, le=10000)
    train_ratio: float = Field(default=.6, gt=0, lt=1, allow_inf_nan=False)
    val_ratio: float = Field(default=.2, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _split(self):
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("Leave a nonzero fraction for the test split.")
        return self


class GeneratorDatasetResponse(StrictModel):
    dataset: DatasetManifest
    test_samples: int


class DatasetSampleCounts(StrictModel):
    dataset_id: str
    version: str
    counts: dict[str, int]
    sample_rate_hz: float | None
    total_samples: int
    guard_samples: int
