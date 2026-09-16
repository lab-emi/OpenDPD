"""Contracts for explicit, illustrative PA simulation and paired dataset creation."""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import Field, model_validator

from .common import Sha256, Slug, StrictModel


class PALocalizedText(StrictModel):
    en: str
    zh: str


class PAParameter(StrictModel):
    key: str
    symbol: str
    symbol_latex: str = ""
    label: PALocalizedText
    description: PALocalizedText
    group: Literal["gain", "memory", "dynamics", "architecture"] = "gain"
    default: float
    minimum: float
    maximum: float
    step: float
    unit: str = ""
    logarithmic: bool = False
    integer: bool = False


class VirtualPAModel(StrictModel):
    model_id: Slug
    category: Literal["reference", "static", "memory", "dynamics", "architecture"]
    name: PALocalizedText
    technology: str
    description: PALocalizedText
    limitations: PALocalizedText
    parameters: list[PAParameter]
    equations: list[str]
    equations_latex: list[str] = Field(default_factory=list)
    references: list[str]


class PAInputDataset(StrictModel):
    signal_id: str = Field(pattern=r"^sg-[a-f0-9]{64}$")
    kind: Literal["pa_input"] = "pa_input"
    name: str
    n_samples: int
    sample_rate_hz: float
    bandwidth_hz: float
    iq_sha256: Sha256
    csv_url: str
    metadata_url: str


class VirtualPARequest(StrictModel):
    input_signal_id: str = Field(pattern=r"^sg-[a-f0-9]{64}$")
    model_id: Slug
    parameters: dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)]] = Field(default_factory=dict, max_length=32)


class PAAnalysis(StrictModel):
    n_samples: int
    duration_ms: float
    input_rms: float
    output_rms: float
    rms_gain_db: float | None
    input_papr_db: float | None
    output_papr_db: float | None
    time_us: list[float]
    input_envelope: list[float]
    output_envelope: list[float]
    am_input: list[float]
    am_output: list[float]
    am_pm_deg: list[float]
    frequency_mhz: list[float]
    input_psd: list[float]
    output_psd: list[float]
    states: dict[str, list[float]]
    notes: list[str]


class VirtualPASimulation(StrictModel):
    simulation_id: str = Field(pattern=r"^vpa-[a-f0-9]{64}$")
    kind: Literal["simulated_pa_output"] = "simulated_pa_output"
    config: VirtualPARequest
    model: VirtualPAModel
    input_iq_sha256: Sha256
    output_iq_sha256: Sha256
    kernel_sha256: Sha256 | None = None
    simulator_source_sha256: Sha256
    sample_rate_hz: float
    analysis: PAAnalysis
    output_csv_url: str
    paired_csv_url: str
    metadata_url: str


class PairedDatasetRequest(StrictModel):
    dataset_id: Slug
    display_name: str = Field(default="Virtual PA · paired input and output", min_length=1, max_length=180)
    guard_samples: int = Field(default=256, ge=0, le=10000)
    train_ratio: float = Field(default=.6, gt=0, lt=1, allow_inf_nan=False)
    val_ratio: float = Field(default=.2, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _split(self):
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("Leave a nonzero fraction for the test split.")
        return self


class VirtualPADatasetRequest(StrictModel):
    input_signal_ids: list[Annotated[str, Field(pattern=r"^sg-[a-f0-9]{64}$")]] = Field(min_length=1, max_length=16)
    model_id: Slug
    parameters: dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)]] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def _unique(self):
        if len(set(self.input_signal_ids)) != len(self.input_signal_ids):
            raise ValueError("Select each PA input only once.")
        return self
