"""Contracts for the NR FR1 standard test models of 3GPP TS 38.141-1 clause 4.9.2.

``NrTestModelConfig`` is a *recipe*: it names a test model and every parameter the
standard leaves open (duplex mode, carriers, subcarrier spacing, channel bandwidth,
cell identity, rank, PN-sequence choices, NB-IoT in-band placement, capture length).
Everything else (resource blocks, FFT size, sample rate, carrier offsets, RBG sets,
boost levels, ...) is *derived* by ``opendpd.core.waveforms.nr_tm.resolve`` and reported
in ``NrTmSummary``. Table-dependent rules (valid bandwidth for a subcarrier spacing,
the 20 ms TDD limit, ...) are enforced by ``opendpd.core.waveforms.nr_tm.validate_recipe``
which the validator below calls lazily, so importing this module stays dependency-free.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import Field, ValidationInfo, model_validator

from .common import StrictModel, is_stored

NrTmModelId = Literal[
    "NR-FR1-TM1.1", "NR-FR1-TM1.2", "NR-FR1-TM2", "NR-FR1-TM2a", "NR-FR1-TM2b",
    "NR-FR1-TM3.1", "NR-FR1-TM3.1a", "NR-FR1-TM3.1b", "NR-FR1-TM3.2", "NR-FR1-TM3.3", "NR-N-TM",
]
NR_TM_MODELS: tuple[str, ...] = get_args(NrTmModelId)
MAX_CARRIERS = 16


class NrTmCarrier(StrictModel):
    """One NR carrier carrying the test model (TS 38.141-1 4.9.2.3: one cell ID per carrier)."""
    scs_khz: Literal[15, 30, 60] = 30
    # Channel bandwidth; N_RB comes from TS 38.104 Table 5.3.2-1 for this subcarrier spacing.
    bandwidth_mhz: int = Field(default=20, ge=3, le=100)
    # None = standard rule: N_cell_ID = 1 for the lowest carrier, 2 for the second lowest, ...
    n_cell_id: int | None = Field(default=None, ge=0, le=1007)
    # Extra spacing (MHz) added to the nominal channel spacing of TS 38.104 5.4.1.2 before this
    # carrier. 0 = contiguous (NRTC1/NRTC2); > 0 starts a new sub-block (non-contiguous, NRTC3).
    # Must be 0 for the first carrier; the resolver enforces the 0.3 MHz grid.
    gap_before_mhz: float = Field(default=0, ge=0, le=1000, allow_inf_nan=False)


class NrTmNbIot(StrictModel):
    """NB-IoT operation in NR in-band (NR-N-TM, TS 38.141-1 4.9.2.2.9): FDD, 15 kHz only."""
    # Which eligible NR RB is punctured: closest to the lower / upper minimum guard band, or both
    # (NRTC1 places a second NB-IoT carrier at the upper edge when more than one is supported).
    position: Literal["lower", "upper", "both"] = "lower"
    # X dB: power boost of the NB-IoT REs over the average RE power (declared >= 6 dB, or >= 3 dB).
    boost_db: float = Field(default=6, ge=3, le=12, allow_inf_nan=False)


class NrTestModelConfig(StrictModel):
    version: Literal["nr-test-model-v1"] = "nr-test-model-v1"
    model: NrTmModelId = "NR-FR1-TM1.1"
    duplex: Literal["fdd", "tdd"] = "fdd"
    carriers: list[NrTmCarrier] = Field(default_factory=lambda: [NrTmCarrier()], min_length=1, max_length=MAX_CARRIERS)
    # Rank-2 transmission (two layers, ports 1000/1001, no precoding) exists only for NR-FR1-TM1.1
    # in the TAE test. The generator exports one stream: ``output_port`` selects the antenna port.
    layers: Literal[1, 2] = 1
    output_port: Literal[0, 1] = 0
    # PN23 handling. "standard" follows the text of 4.9.2.3.2 (one stream per n_RNTI user, restarted
    # in the first allocated slot of every frame). "matlab" reproduces MATLAB 5G Toolbox
    # (one stream per PDSCH allocation object, never restarted): both are allowed readings.
    data_convention: Literal["standard", "matlab"] = "standard"
    # TDD only (the standard: "all symbols (DL, UL or special slots) or only DL symbols").
    pn_scope: Literal["dl_symbols", "all_symbols"] = "dl_symbols"
    nb_iot: NrTmNbIot | None = None
    # None = automatic: the largest of 1/2/4 such that the standard duration fits 1,000,000 samples
    # (falls back to 1 when even that does not fit).
    oversampling: Literal[1, 2, 4, 8] | None = None
    # Capture length in subframes (1 ms). None = standard duration (10 FDD / 20 TDD), shortened to
    # the largest whole number of subframes that fits 1,000,000 samples; the summary says so.
    subframes: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def _table_rules(self, info: ValidationInfo):
        if not is_stored(info):      # a recipe stored by an earlier release is not judged by the table rules of this one
            from opendpd.core.waveforms.nr_tm import validate_recipe  # lazy: schemas stay light
            validate_recipe(self)
        return self


class NrTmUser(StrictModel):
    """One user (n_RNTI) of the test model in a reference full-DL slot."""
    rnti: int
    role: str
    modulation: str
    prbs: int
    # Inclusive [first, last] PRB ranges. TM2/2a/2b list the three slot-class positions (3n, 3n+1, 3n+2).
    rb_ranges: list[list[int]]
    start_symbol: int
    level_db: float
    evm_measured: bool = False


class NrTmCarrierSummary(StrictModel):
    index: int
    scs_khz: int
    bandwidth_mhz: int
    nrb: int
    subcarriers: int
    n_fft: int
    n_cell_id: int
    center_offset_hz: float
    occupied_bandwidth_hz: float
    slots: int
    slots_per_frame: int
    rbg_size: int | None = None
    n_rbg: int | None = None
    boosted_rbgs: list[int] = Field(default_factory=list)
    users: list[NrTmUser] = Field(default_factory=list)


class NrTmSummary(StrictModel):
    model: NrTmModelId
    carriers: list[NrTmCarrierSummary]
    sample_rate_hz: float
    native_sample_rate_hz: float
    oversampling: int
    subframes: int
    standard_subframes: int
    truncated: bool
    n_samples: int
    duration_ms: float
    aggregated_bandwidth_hz: float
    # One entry per slot of one 5 ms pattern of the first carrier ("D", "U", "S 10D/2U", "S 12D", ...).
    tdd_pattern: list[str] | None = None
    nb_iot_rbs: list[int] | None = None
    warnings: list[str] = Field(default_factory=list)
