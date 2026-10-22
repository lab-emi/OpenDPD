"""Request/response contracts of the NR test-model endpoints (catalog and resolve)."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import StrictModel
from .nr_test_model import NrTestModelConfig, NrTmModelId, NrTmSummary
from .signal_generator import GeneratorConfig


class NrTmFixedParameter(StrictModel):
    """A parameter the standard fixes for this model (shown read-only), e.g. 'PDCCH: 1 CCE, RB 0-2, symbols 0-1'."""
    name: str
    value: str


class NrTmUserInfo(StrictModel):
    rnti: int
    role: str
    modulation: str
    evm_measured: bool = False


class NrTmModelInfo(StrictModel):
    model: NrTmModelId
    clause: str
    title: str
    description: str
    used_for: list[str]
    modulations: list[str]
    users: list[NrTmUserInfo]
    fixed_parameters: list[NrTmFixedParameter]
    supports_layers: bool = False
    nb_iot: bool = False
    fdd_only: bool = False
    scs_khz: list[int]


class NrTmTddPattern(StrictModel):
    """TS 38.141-1 Table 4.9.2.2-1 (TS 38.331 TDD-UL-DL-ConfigCommon semantics)."""
    scs_khz: int
    period_ms: float
    slots_per_period: int
    dl_slots: int
    dl_symbols: int
    ul_slots: int
    ul_symbols: int
    slot_map: list[str]


class NrTmNbIotInfo(StrictModel):
    scs_khz: int
    bandwidths_mhz: list[int]
    # Lowest / highest eligible NR RB per channel bandwidth (key = MHz as text).
    lower_rb: dict[str, int]
    upper_rb: dict[str, int]
    eligible_rbs: dict[str, list[int]]


class NrTmLimits(StrictModel):
    max_carriers: int
    max_samples: int
    max_cell_id: int
    standard_subframes_fdd: int
    standard_subframes_tdd: int
    carrier_gap_step_mhz: float


class NrTmCatalog(StrictModel):
    version: Literal["nr-test-model-catalog-v1"] = "nr-test-model-catalog-v1"
    standard: str
    models: list[NrTmModelInfo]
    scs_khz: list[int]
    # TS 38.104 Table 5.3.2-1: scs (text) -> channel bandwidth in MHz (text) -> N_RB.
    resource_blocks: dict[str, dict[str, int]]
    tdd_patterns: list[NrTmTddPattern]
    nb_iot: NrTmNbIotInfo
    limits: NrTmLimits
    default_recipe: NrTestModelConfig
    default_config: GeneratorConfig
    notes: list[str] = Field(default_factory=list)


class NrTmResolution(StrictModel):
    """A recipe plus the complete, validated generator configuration derived from it."""
    recipe: NrTestModelConfig
    config: GeneratorConfig
    summary: NrTmSummary
