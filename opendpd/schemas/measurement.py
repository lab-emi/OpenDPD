"""Measured DPD evidence (plan S16): a physical PA driven by an exported signal.

The operator plays the pre-distorted input ``u`` exported by a ``run_dpd`` run
(and, under the same conditions, the target input ``x``) and captures the PA
output. OpenDPD aligns each capture to what was played, scores it as captured
and records every condition the operator declares. Nothing in this module is
measured by OpenDPD itself: the attestation string travels with every result.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from .common import Sha256, Slug, StrictModel

ATTESTATION = "user-provided measurement; not independently verified by OpenDPD"
MOCK_ATTESTATION = "mock instrument adapter: a synthetic stand-in for a PA, not a measurement"


class CaptureRef(StrictModel):
    """A capture file under ``<workspace>/imports`` (an upload or a copied file); never an arbitrary path."""

    path: str = Field(min_length=1, max_length=512)          # relative to the imports directory, POSIX separators
    sha256: Optional[Sha256] = None                          # bound at submission, verified again when the run executes
    columns: Optional[Tuple[str, str]] = None                # (I, Q) CSV columns or .npz keys; default: I/Q or I_out/Q_out
    declared_output_power_dbm: Optional[float] = None        # the operator's reading for this capture


class MeasurementConditions(StrictModel):
    """What the operator declares about the set-up. Recorded verbatim, never inferred."""

    pa: str = Field(min_length=1, max_length=200)             # device under test
    capture_chain: str = Field(min_length=1, max_length=500)  # generator → PA → attenuation → analyser, as wired
    sample_rate_hz: float = Field(gt=0)                       # rate of the captured files
    drive: str = Field(min_length=1, max_length=200)          # generator level / PA input drive as set
    gain_db: Optional[float] = None                           # PA gain as declared
    calibration: str = Field(default="none", min_length=1, max_length=500)
    measured_at: datetime
    temperature_c: Optional[float] = None
    operator: Optional[str] = Field(default=None, max_length=100)
    notes: Optional[str] = Field(default=None, max_length=2000)


class MeasurementConfig(StrictModel):
    """The ``evaluate_measured`` task: which exported signal was played and what came back."""

    apply_run_id: Slug                                        # the run_dpd run whose dpd-output was played
    played_sha256: Optional[Sha256] = None                    # hash of that artifact, bound at submission
    with_dpd: CaptureRef                                      # PA output while u = DPD(x) was played
    without_dpd: Optional[CaptureRef] = None                  # PA output while x was played under the same conditions
    conditions: MeasurementConditions
    source: Literal["manual", "mock_adapter"] = "manual"
    playback: Literal["loop", "single"] = "loop"              # loop: the file repeated, so a wrapped window is valid


class CaptureAlignment(StrictModel):
    """How one capture was aligned to the signal that was played, in capture units."""

    role: Literal["with_dpd", "without_dpd"]
    artifact_id: Slug                                         # the copy in this run's artifact manifest
    raw_sha256: Sha256                                        # of the file the operator provided
    n_samples_raw: int = Field(ge=1)
    sample_rate_hz: float = Field(gt=0)
    resample_ratio: Optional[Tuple[int, int]] = None          # (up, down) to the dataset rate; None: same rate
    delay_samples: int = Field(ge=0)
    wrapped: bool = False                                     # the window wrapped around the file (looped playback)
    correlation: float = Field(ge=0, le=1)                    # |<played, aligned>| / (|played| |aligned|)
    gain_abs: float = Field(ge=0)                             # |g|, least-squares gain of the aligned capture onto x
    gain_db: float
    gain_phase_deg: float
    rms: float = Field(ge=0)
    peak_abs: float = Field(ge=0)
    declared_output_power_dbm: Optional[float] = None


class MeasurementEvidence(StrictModel):
    """Stored as ``measurement.json`` in the run and embedded in every result of the run."""

    attestation: str = Field(min_length=1)
    apply_run_id: Slug
    played_sha256: Sha256                                     # the export's hash; its copy is artifact played-signal
    conditions: MeasurementConditions
    captures: List[CaptureAlignment] = Field(min_length=1, max_length=2)
    level_difference_db: Optional[float] = None               # 20 log10(rms with DPD / rms without DPD), capture units
    declared_power_difference_db: Optional[float] = None      # declared with-DPD minus without-DPD output power

    @model_validator(mode="after")
    def _roles(self) -> "MeasurementEvidence":
        roles = [c.role for c in self.captures]
        if roles[0] != "with_dpd" or len(roles) != len(set(roles)):
            raise ValueError("captures list the with_dpd alignment first and each role once")
        return self
