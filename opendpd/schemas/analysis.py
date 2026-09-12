"""Read-only dataset inspection, shared by the CLI, API and Studio.

These are presentation contracts for existing core computations, not new
metric definitions or evaluation profiles.
"""

from typing import List, Literal, Optional, Tuple

from pydantic import Field

from .common import StrictModel
from .diagnostics import DiagnosticReport


class InspectionReading(StrictModel):
    value: Optional[float] = None
    status: Literal["ok", "invalid", "not_applicable", "missing_reference", "review_required"]
    reason: Optional[str] = None


class InspectionMeasurement(StrictModel):
    name: str
    label: str
    unit: str
    method: str
    input: Optional[InspectionReading] = None
    output: InspectionReading


class SpectrumTrace(StrictModel):
    name: str
    role: str
    psd_db: List[float]


class SpectrumBands(StrictModel):
    main: Tuple[float, float]
    adjacent: List[Tuple[float, float]]


class InspectionSpectrum(StrictModel):
    version: str
    kind: Literal["spectrum"]
    axis: Literal["hz", "normalized"]
    sample_rate_hz: Optional[float]
    nperseg: int
    n_samples: int
    frequency: List[float]
    traces: List[SpectrumTrace]
    bands: Optional[SpectrumBands]
    estimator: str


class IQTrace(StrictModel):
    name: str
    role: str
    i: List[float]
    q: List[float]


class InspectionTime(StrictModel):
    version: str
    kind: Literal["time"]
    start: int
    n: int
    n_samples: int
    traces: List[IQTrace]


class InspectionIQ(StrictModel):
    version: str
    kind: Literal["iq"]
    mode: Literal["samples"]
    stride: int
    n_samples: int
    traces: List[IQTrace]


class ConstellationTrace(IQTrace):
    n_symbols: int
    stride: int
    equalized: bool


class InspectionConstellation(StrictModel):
    status: Literal["ok", "unavailable", "invalid"]
    reason: Optional[str] = None
    dataset_name: Optional[str] = None
    modulation: Optional[str] = None
    demodulator: Optional[str] = None
    sample_range: Optional[Tuple[int, int]] = None
    source_sample_range: Optional[Tuple[int, int]] = None
    fft_size: Optional[int] = None
    active_subcarriers_per_carrier: Optional[int] = None
    n_carriers: Optional[int] = None
    traces: List[ConstellationTrace] = Field(default_factory=list)
    note: str


class AmTrace(StrictModel):
    name: str
    role: str
    amp_out: List[float]
    phase_deg: List[float]


class InspectionAm(StrictModel):
    version: str
    kind: Literal["am"]
    stride: int
    n_points: int
    n_samples: int
    amp_in: List[float]
    traces: List[AmTrace]
    note: str


class DatasetAnalysis(StrictModel):
    version: Literal["dataset-inspection-v1"] = "dataset-inspection-v1"
    dataset_id: str
    data_version: str
    total_samples: int
    sample_range: Tuple[int, int]
    metadata_complete: bool
    inspection_ready: bool
    diagnostics: DiagnosticReport
    measurements: List[InspectionMeasurement]
    spectrum: Optional[InspectionSpectrum] = None
    time: Optional[InspectionTime] = None
    iq: Optional[InspectionIQ] = None
    constellation: Optional[InspectionConstellation] = None
    am: Optional[InspectionAm] = None
    notes: List[str] = Field(default_factory=list)
