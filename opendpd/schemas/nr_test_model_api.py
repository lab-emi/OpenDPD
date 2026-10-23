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
    # True when Studio generates the model for FDD only. For NR-N-TM that is a limit of Studio, not of the standard: TS 38.141-1
    # Table 4.9.2.2-1a also defines a TDD form of the model, which is not implemented in this release.
    fdd_only: bool = Field(default=False, description=(
        "True when Studio generates this model for FDD only. NR-N-TM: the standard also defines a TDD form (TS 38.141-1 Table 4.9.2.2-1a), "
        "which Studio does not implement in this release."))
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
    # Smallest NB-IoT power boost X (dB) per host channel bandwidth (key = MHz as text) at the positions Studio offers (lower_rb, upper_rb):
    # TS 38.104 Table 6.3.4.2-1, 6 dB up to 20 MHz and 3 dB from 25 MHz. The resolver refuses a smaller boost_db.
    min_boost_db: dict[str, float] = Field(description=(
        "Smallest NB-IoT boost X in dB per host channel bandwidth (MHz as text) at the positions lower_rb and upper_rb "
        "(TS 38.104 Table 6.3.4.2-1); a smaller boost_db is refused."))
    max_boost_db: float = Field(description="Largest NB-IoT boost X in dB that Studio accepts. A Studio limit: the standard sets no upper bound.")


class NrTmLimits(StrictModel):
    max_carriers: int = Field(description="Largest number of carriers of a recipe. A Studio limit: the standard sets none.")
    max_samples: int = Field(description="Largest number of complex samples of a capture (a Studio limit); the capture is shortened or refused above it.")
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
