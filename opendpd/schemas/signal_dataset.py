"""Named collections reference immutable waveforms; each member keeps its own rate."""
from typing import Annotated, Literal

from pydantic import Field

from .common import Slug, StrictModel
from .signal_analyzer import AnalyzerSourceInfo

InputDatasetName = Annotated[str, Field(pattern=r"^syn_pa_in_[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)]
PairedDatasetName = Annotated[str, Field(pattern=r"^syn_pa_inout_[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)]


class AnalyzerDataset(StrictModel):
    dataset_id: Slug
    name: str
    kind: Literal["pa_input", "pa_output", "paired", "upload"]
    signals: list[AnalyzerSourceInfo]
    download_url: str | None = None


class SignalDataset(AnalyzerDataset):
    dataset_id: str = Field(pattern=r"^sds-[a-f0-9]{64}$")
    kind: Literal["pa_input", "pa_output"]
    requested_name: str
    signals: list[AnalyzerSourceInfo] = Field(min_length=1, max_length=16)
