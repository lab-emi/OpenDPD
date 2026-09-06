"""One supervised capture session: arm, play, capture, RF off, record (plan S16)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np

from opendpd.core.measurement import to_complex, to_iq

from .base import Instrument, SafetyLimits
from .mock import MockInstrument
from .safety import Interlock

ADAPTERS: Dict[str, Type[Instrument]] = {"mock": MockInstrument}


def list_adapters():
    return [cls.info for cls in ADAPTERS.values()]


CAPTURE_MARGIN = 4096            # samples captured beyond one period, so the delayed period is complete


def run_capture_session(instrument: Instrument, played: np.ndarray, sample_rate_hz: float, *, operator: str,
                        out: Path, limits: Optional[SafetyLimits] = None,
                        requested_power_dbm: Optional[float] = None) -> Path:
    """Play ``played`` (I/Q ``(n, 2)`` or complex) and capture one period plus a margin into ``out`` (``.npy``,
    ``(m, 2)`` float32) with a ``<out>.session.json`` record. RF goes off before this returns, on every path."""
    z = to_complex(played)
    n = int(z.size + CAPTURE_MARGIN)
    out = Path(out)
    record = {"adapter": instrument.info.model_dump(mode="json"), "operator": operator,
              "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "sample_rate_hz": float(sample_rate_hz), "played_sha256": hashlib.sha256(to_iq(z).tobytes()).hexdigest(),
              "n_played": int(z.size), "n_requested": n, "requested_power_dbm": requested_power_dbm,
              "mock": instrument.info.kind == "mock"}
    with Interlock(instrument, limits) as lock:
        try:
            lock.arm(operator)
            lock.play(z, sample_rate_hz, requested_power_dbm=requested_power_dbm)
            captured = lock.capture(n, sample_rate_hz)
        finally:
            lock.disarm()
            record["interlock"] = {"final_state": lock.state, "trip_reason": lock.trip_reason, "log": lock.log,
                                   "limits": lock.limits.model_dump()}
    iq = to_iq(captured)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, iq)
    record["capture_sha256"] = hashlib.sha256(iq.tobytes()).hexdigest()
    record["n_captured"] = int(iq.shape[0])
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(str(out) + ".session.json").write_text(json.dumps(record, indent=2))
    return out
