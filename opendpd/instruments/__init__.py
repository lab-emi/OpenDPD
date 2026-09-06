"""Instrument adapters with fail-closed safety (plan S16). Only the mock ships; see docs/architecture/instruments.md."""

from .base import AdapterInfo, Instrument, InstrumentError, LinkLost, SafetyLimits, SafetyViolation
from .mock import MockInstrument
from .safety import ALLOW_RF_ENV, Interlock
from .session import ADAPTERS, list_adapters, run_capture_session

__all__ = ["ADAPTERS", "ALLOW_RF_ENV", "AdapterInfo", "Instrument", "InstrumentError", "Interlock", "LinkLost",
           "MockInstrument", "SafetyLimits", "SafetyViolation", "list_adapters", "run_capture_session"]
