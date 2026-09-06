"""Fail-closed control of RF output (plan S16).

RF output is off until a person arms the session; every abnormal path (a
limit, a timeout, a lost link, an exception, the end of the ``with`` block)
turns it off again and leaves the interlock tripped. Real adapters
additionally need ``OPENDPD_ALLOW_RF_OUTPUT=1`` in the environment of the
process, which continuous integration never sets and which no agent may set on
its own (AGENTS.md); the mock adapter emits nothing and still follows the same
arming rule so the procedure is exercised end to end.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base import Instrument, InstrumentError, LinkLost, SafetyLimits, SafetyViolation

ALLOW_RF_ENV = "OPENDPD_ALLOW_RF_OUTPUT"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Interlock:
    """States: ``disarmed`` (RF off) → ``armed`` (by a named operator) → ``active`` (during play/capture) →
    ``disarmed``; any failure lands in ``tripped`` with RF off, and a tripped interlock cannot be re-armed."""

    def __init__(self, instrument: Instrument, limits: Optional[SafetyLimits] = None) -> None:
        self.instrument = instrument
        self.limits = limits or instrument.info.default_limits
        self.state = "disarmed"
        self.operator: Optional[str] = None
        self.trip_reason: Optional[str] = None
        self.log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._record("created", limits=self.limits.model_dump())

    # --- bookkeeping -------------------------------------------------------------------

    def _record(self, event: str, **fields: Any) -> None:
        self.log.append({"at": _now(), "event": event, "state": self.state, **fields})

    def _rf_off(self, reason: str) -> None:
        try:
            self.instrument.rf_off()
        finally:
            self._record("rf_off", reason=reason)

    def trip(self, reason: str) -> None:
        """RF off, state ``tripped``; idempotent."""
        with self._lock:
            self._rf_off(reason)
            self.state = "tripped"
            self.trip_reason = self.trip_reason or reason
            self._record("tripped", reason=reason)

    # --- arming --------------------------------------------------------------------------

    def arm(self, operator: str) -> None:
        """A named person arms the session. Real adapters also need the environment gate."""
        if self.state == "tripped":
            raise SafetyViolation(f"interlock tripped ({self.trip_reason}); start a new session")
        if self.state != "disarmed":
            raise SafetyViolation(f"cannot arm from state {self.state}")
        if not operator or not operator.strip():
            raise SafetyViolation("arming needs the operator's name; RF output stays off")
        if self.instrument.info.rf_output_capable and os.environ.get(ALLOW_RF_ENV) != "1":
            raise SafetyViolation(f"real RF output is not permitted in this environment ({ALLOW_RF_ENV} is not "
                                      "'1'); only an approved laboratory session may set it")
        self.instrument.heartbeat()
        self.operator = operator.strip()
        self.state = "armed"
        self._record("armed", operator=self.operator)

    def disarm(self) -> None:
        with self._lock:
            self._rf_off("disarm")
            if self.state != "tripped":
                self.state = "disarmed"
            self._record("disarmed")

    # --- guarded operations --------------------------------------------------------------

    def _guard(self, what: str, fn: Callable[[], Any]) -> Any:
        if self.state != "armed":
            raise SafetyViolation(f"{what} requires an armed interlock (state {self.state})")
        self.state = "active"
        self._record(what)
        result: Dict[str, Any] = {}
        error: List[BaseException] = []

        def body() -> None:
            try:
                result["value"] = fn()
            except BaseException as err:  # noqa: BLE001 - re-raised on the calling thread
                error.append(err)

        worker = threading.Thread(target=body, name=f"instrument-{what}", daemon=True)
        started = time.monotonic()
        worker.start()
        last_beat = started
        while worker.is_alive():
            worker.join(0.05)
            now = time.monotonic()
            if now - started > self.limits.timeout_s:
                self.trip(f"{what} exceeded {self.limits.timeout_s:g} s")
                raise InstrumentError(f"{what} timed out after {self.limits.timeout_s:g} s; RF output off")
            if now - last_beat >= min(0.5, self.limits.link_timeout_s / 2):
                last_beat = now
                try:
                    self.instrument.heartbeat()
                except LinkLost as err:
                    self.trip(f"link lost during {what}: {err}")
                    raise
        if error:
            self.trip(f"{what} failed: {error[0]}")
            raise error[0]
        self.state = "armed"
        self._record(f"{what}_done", seconds=round(time.monotonic() - started, 3))
        return result["value"]

    def play(self, signal: np.ndarray, sample_rate_hz: float, *, requested_power_dbm: Optional[float] = None) -> None:
        """Checks the limits, then lets the adapter start the output."""
        z = np.asarray(signal)
        peak = float(np.max(np.abs(z))) if z.size else 0.0
        if z.size == 0 or not np.all(np.isfinite(z)):
            raise SafetyViolation("refusing to play an empty or non-finite signal")
        if peak > self.limits.max_peak_abs:
            raise SafetyViolation(f"peak |signal| {peak:.4g} exceeds the limit {self.limits.max_peak_abs:g}; nothing sent")
        ceiling = self.limits.max_output_power_dbm
        if ceiling is not None and requested_power_dbm is not None and requested_power_dbm > ceiling:
            raise SafetyViolation(f"requested {requested_power_dbm:g} dBm exceeds the ceiling {ceiling:g} dBm; nothing sent")
        self._guard("play", lambda: self.instrument.play(z, sample_rate_hz))

    def capture(self, n_samples: int, sample_rate_hz: float) -> np.ndarray:
        return self._guard("capture", lambda: self.instrument.capture(n_samples, sample_rate_hz))

    def abort(self, reason: str = "abort requested") -> None:
        """Manual stop: RF off and tripped, so nothing runs afterwards."""
        self.trip(reason)

    def __enter__(self) -> "Interlock":
        self.instrument.connect()
        self._record("connected")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and self.state != "tripped":
            self.trip(f"session ended with {type(exc).__name__}: {exc}")
        self.disarm()
        try:
            self.instrument.disconnect()
        finally:
            self._record("disconnected")
