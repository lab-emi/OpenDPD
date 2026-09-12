"""The instrument adapter contract (plan S16).

An adapter drives one generator/analyser chain: it plays a baseband signal and
captures the PA output. Adapters never decide when RF may be on: the
``Interlock`` in ``safety.py`` does, and every adapter must make ``rf_off``
idempotent and safe to call at any moment, because that is what runs on every
abnormal path.
"""

from __future__ import annotations

import abc
from typing import Literal, Optional

import numpy as np
from pydantic import Field

from opendpd.schemas.common import StrictModel


class InstrumentError(RuntimeError):
    """The adapter could not do what was asked; RF output is off when this propagates."""


class LinkLost(InstrumentError):
    """The instrument stopped answering."""


class SafetyViolation(InstrumentError):
    """A limit or the arming rule would have been violated; nothing was sent."""


class SafetyLimits(StrictModel):
    """Limits the operator declares before arming. The interlock enforces them; adapters may add stricter ones."""

    max_peak_abs: float = Field(default=1.0, gt=0)              # digital full scale of the played samples
    max_output_power_dbm: Optional[float] = None                # operator's ceiling for the requested output power
    timeout_s: float = Field(default=30.0, gt=0)                # any play/capture longer than this trips the interlock
    link_timeout_s: float = Field(default=5.0, gt=0)            # silence longer than this is a loss of link


class AdapterInfo(StrictModel):
    adapter_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    description: str = Field(min_length=1)
    kind: Literal["mock", "real"]
    rf_output_capable: bool                                     # a mock never emits RF
    default_limits: SafetyLimits = Field(default_factory=SafetyLimits)


class Instrument(abc.ABC):
    """One generator + analyser chain. Methods raise ``InstrumentError`` (or a subclass) on failure."""

    info: AdapterInfo

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def heartbeat(self) -> None:
        """Raise ``LinkLost`` when the instrument does not answer."""

    @abc.abstractmethod
    def rf_off(self) -> None:
        """Turn the output off. Idempotent; must never raise."""

    @abc.abstractmethod
    def play(self, signal: np.ndarray, sample_rate_hz: float) -> None:
        """Start looping ``signal`` (complex baseband, digital full scale 1.0). Only the interlock calls this."""

    @abc.abstractmethod
    def capture(self, n_samples: int, sample_rate_hz: float) -> np.ndarray:
        """Return ``n_samples`` of the analyser's complex baseband output."""
