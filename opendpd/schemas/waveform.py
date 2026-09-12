"""Reference waveform specifications (plan S15).

A waveform is fully described by its specification: the numerology is fixed
per ``waveform_id`` and only the seed and the length vary, so the reference
symbols and the time-domain signal can always be regenerated instead of
shipped. A dataset that was captured while an instrument played a waveform
can be *bound* to it; that binding is what allows a data-aided (known-symbol)
evaluation of a PA or DPD output.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from opendpd.schemas.common import Sha256, StrictModel

# The one configuration supported in this version: CP-OFDM with the LTE 20 MHz numerology.
OFDM_LTE20_V1 = "ofdm-lte20-v1"
OFDM_LTE20_SAMPLE_RATE_HZ = 30_720_000.0
OFDM_LTE20_SUBCARRIER_SPACING_HZ = 15_000.0
OFDM_LTE20_FFT_SIZE = 2048
OFDM_LTE20_OCCUPIED_SUBCARRIERS = 1200          # 100 resource blocks of 12; the DC subcarrier stays empty
OFDM_LTE20_SAMPLES_PER_SUBFRAME = 30_720        # 1 ms: 2 slots of 7 symbols with the normal cyclic prefix


class WaveformSpec(StrictModel):
    """Everything needed to regenerate a reference waveform bit for bit."""

    waveform_id: Literal["ofdm-lte20-v1"] = OFDM_LTE20_V1
    version: int = Field(default=1, ge=1)
    sample_rate_hz: float = OFDM_LTE20_SAMPLE_RATE_HZ
    subcarrier_spacing_hz: float = OFDM_LTE20_SUBCARRIER_SPACING_HZ
    fft_size: int = OFDM_LTE20_FFT_SIZE
    occupied_subcarriers: int = OFDM_LTE20_OCCUPIED_SUBCARRIERS
    cyclic_prefix: Literal["normal"] = "normal"
    modulation: Literal["64QAM"] = "64QAM"
    n_subframes: int = Field(default=10, ge=1, le=1000)   # 10 subframes = one 10 ms frame
    seed: int = Field(default=0, ge=0, le=2 ** 32 - 1)

    @model_validator(mode="after")
    def _fixed_numerology(self) -> "WaveformSpec":
        fixed = {
            "sample_rate_hz": OFDM_LTE20_SAMPLE_RATE_HZ, "subcarrier_spacing_hz": OFDM_LTE20_SUBCARRIER_SPACING_HZ,
            "fft_size": OFDM_LTE20_FFT_SIZE, "occupied_subcarriers": OFDM_LTE20_OCCUPIED_SUBCARRIERS, "version": 1,
        }
        wrong = [f"{k}={getattr(self, k)!r} (must be {v!r})" for k, v in fixed.items() if getattr(self, k) != v]
        if wrong:
            raise ValueError(f"{self.waveform_id} has a fixed numerology; " + ", ".join(wrong))
        return self

    @property
    def n_samples(self) -> int:
        return self.n_subframes * OFDM_LTE20_SAMPLES_PER_SUBFRAME

    @property
    def n_symbols(self) -> int:
        return self.n_subframes * 14


class WaveformBinding(StrictModel):
    """A dataset's link to the waveform its input column was captured from.

    ``input_offset_samples`` is the position of the dataset's first raw input sample inside one period of the
    waveform (instruments play a waveform in a loop); ``correlation`` is the normalised cross-correlation peak
    that established it. The binding never replaces the captured data: it only says which symbols were sent.
    """

    spec: WaveformSpec
    input_offset_samples: int = Field(ge=0)
    correlation: float = Field(ge=0.0, le=1.0)
    input_sample_rate_hz: float = Field(gt=0)
    package_sha256: Optional[Sha256] = None
    bound_at: Optional[str] = None
