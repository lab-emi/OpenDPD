"""A dry-run adapter: no hardware, no RF (plan S16).

It behaves like a generator/analyser pair around a fixed synthetic PA so the
whole export → play → capture → import → evaluate procedure runs in tests and
continuous integration. Results that come from it are marked ``is_mock`` and
say so in their attestation; they are never evidence about a PA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AdapterInfo, Instrument, InstrumentError, LinkLost, SafetyLimits

MOCK_DELAY_SAMPLES = 37
MOCK_GAIN = 2.0 * np.exp(1j * np.deg2rad(30.0))
MOCK_CUBIC = 0.12                # y = g (v - c |v|^2 v): a memoryless compression, nothing like a real PA's memory
MOCK_NOISE_RMS = 1e-3


class MockInstrument(Instrument):
    info = AdapterInfo(adapter_id="mock", kind="mock", rf_output_capable=False,
                       description="dry-run generator/analyser around a fixed cubic stand-in PA; emits nothing",
                       default_limits=SafetyLimits(max_peak_abs=1.0, timeout_s=30.0, link_timeout_s=5.0))

    def __init__(self, *, seed: int = 0, lose_link_after: Optional[int] = None, fail_capture: bool = False,
                 capture_delay_s: float = 0.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.connected = False
        self.output_on = False
        self.playing: Optional[np.ndarray] = None
        self.rf_off_calls = 0
        self._beats = 0
        self._lose_link_after = lose_link_after
        self._fail_capture = fail_capture
        self._capture_delay_s = capture_delay_s

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.output_on = False
        self.connected = False

    def heartbeat(self) -> None:
        self._beats += 1
        if self._lose_link_after is not None and self._beats > self._lose_link_after:
            raise LinkLost("mock instrument stopped answering")

    def rf_off(self) -> None:
        self.rf_off_calls += 1
        self.output_on = False
        self.playing = None

    def play(self, signal: np.ndarray, sample_rate_hz: float) -> None:
        if not self.connected:
            raise InstrumentError("not connected")
        self.playing = np.asarray(signal, dtype=np.complex128).reshape(-1)
        self.output_on = True

    def capture(self, n_samples: int, sample_rate_hz: float) -> np.ndarray:
        if self._capture_delay_s:
            import time
            time.sleep(self._capture_delay_s)
        if self._fail_capture:
            raise InstrumentError("mock analyser overload")
        if not self.output_on or self.playing is None:
            raise InstrumentError("nothing is playing")
        v = self.playing
        reps = -(-(n_samples + MOCK_DELAY_SAMPLES) // v.size) + 1
        stream = np.tile(v, reps)
        y = MOCK_GAIN * (stream - MOCK_CUBIC * np.abs(stream) ** 2 * stream)
        y = np.roll(y, MOCK_DELAY_SAMPLES)[:n_samples]
        noise = self.rng.normal(size=n_samples) + 1j * self.rng.normal(size=n_samples)
        return y + MOCK_NOISE_RMS / np.sqrt(2) * noise
