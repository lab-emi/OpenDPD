"""Contracts for the NR FR1 standard test models of 3GPP TS 38.141-1 clause 4.9.2.

``NrTestModelConfig`` is a *recipe*: it names a test model and the parameters Studio
lets you vary (duplex mode, carriers, subcarrier spacing, channel bandwidth, rank,
PN-sequence choices, NB-IoT in-band placement and boost, capture length, and a cell
identity per carrier, which is a Studio extension: the standard fixes N_ID^cell = n).
Everything else (resource blocks, FFT size, sample rate, carrier offsets, RBG sets,
boost levels, ...) is *derived* by ``opendpd.core.waveforms.nr_tm.resolve`` and reported
in ``NrTmSummary``. Table-dependent rules (valid bandwidth for a subcarrier spacing,
the 20 ms TDD limit, ...) are enforced by ``opendpd.core.waveforms.nr_tm.validate_recipe``
which the validator below calls lazily, so importing this module stays dependency-free.
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Literal, get_args

from pydantic import Field, ValidationInfo, field_validator, model_validator

from .common import StrictModel, is_stored

NrTmModelId = Literal[
    "NR-FR1-TM1.1", "NR-FR1-TM1.2", "NR-FR1-TM2", "NR-FR1-TM2a", "NR-FR1-TM2b",
    "NR-FR1-TM3.1", "NR-FR1-TM3.1a", "NR-FR1-TM3.1b", "NR-FR1-TM3.2", "NR-FR1-TM3.3", "NR-N-TM",
]
NR_TM_MODELS: tuple[str, ...] = get_args(NrTmModelId)
MAX_CARRIERS = 16
# Studio limits, not limits of the standard (TS 38.141-1 sets neither a number of carriers nor an upper bound for X).
MAX_NB_IOT_BOOST_DB = 12
# Sub-block gaps are multiples of 0.3 MHz, the least common multiple of the 100 kHz channel raster and the 15/30/60 kHz subcarrier
# spacings (nr_tm.tables.GAP_STEP_KHZ / 1000); a value is stored as the exact decimal multiple so that one gap has one recipe and one id.
GAP_STEP_MHZ = Fraction(3, 10)
GAP_TOLERANCE_MHZ = 1e-9          # float noise such as 0.8999999999999999 is accepted and snapped; anything farther off the grid is refused


class NrTmCarrier(StrictModel):
    """One NR carrier carrying the test model (TS 38.141-1 4.9.2.3: one cell ID per carrier)."""
    scs_khz: Literal[15, 30, 60] = 30
    # Channel bandwidth; N_RB comes from TS 38.104 Table 5.3.2-1 for this subcarrier spacing.
    bandwidth_mhz: int = Field(default=20, ge=3, le=100)
    # None = standard rule: N_cell_ID = 1 for the lowest carrier, 2 for the second lowest, ... Any other value is a Studio
    # extension (4.9.2.3 fixes N_ID^cell = n) and is reported in the warnings of the summary.
    n_cell_id: int | None = Field(default=None, ge=0, le=1007)
    # Extra spacing (MHz) added to the nominal channel spacing of TS 38.104 5.4.1.2 before this
    # carrier. 0 = contiguous (NRTC1/NRTC2); > 0 starts a new sub-block (non-contiguous, NRTC3).
    # Must be 0 for the first carrier. A multiple of 0.3 MHz; the value is stored as that exact multiple (and -0.0 as 0.0).
    gap_before_mhz: float = Field(default=0, ge=0, le=1000, allow_inf_nan=False, description=(
        "Extra spacing in MHz added to the nominal channel spacing before this carrier (a non-contiguous sub-block gap). 0 = contiguous. "
        "Must be 0 for the first carrier and a multiple of 0.3 MHz; a value within 1e-9 MHz of such a multiple is stored as that exact multiple."))

    @field_validator("gap_before_mhz")
    @classmethod
    def _gap_on_the_grid(cls, value: float, info: ValidationInfo) -> float:
        exact = float(GAP_STEP_MHZ * round(value / float(GAP_STEP_MHZ)))
        if abs(value - exact) <= GAP_TOLERANCE_MHZ:
            return exact
        if is_stored(info):               # a stored recipe only has to be well-formed
            return value + 0.0
        raise ValueError(f"the gap before a carrier must be a multiple of 0.3 MHz (the least common multiple of the 100 kHz channel raster "
                         f"and the subcarrier spacings), not {value:g} MHz.")


class NrTmNbIot(StrictModel):
    """NB-IoT operation in NR in-band (NR-N-TM, TS 38.141-1 4.9.2.2.9): Studio generates FDD with 15 kHz hosts only."""
    # Which eligible NR RB is punctured: closest to the lower / upper minimum guard band, or both
    # (NRTC1 places a second NB-IoT carrier at the upper edge when more than one is supported).
    position: Literal["lower", "upper", "both"] = "lower"
    # X dB: power boost of the NB-IoT REs over the average RE power, the manufacturer's declared maximum NB-IoT power dynamic range
    # (TS 38.141-1 D.43, TS 38.104 Table 6.3.4.2-1: at least +6 dB, or +3 dB for the outermost positions of 25 MHz and wider hosts).
    # The resolver enforces the minimum of the host bandwidth; 12 dB is a Studio limit, the standard has no upper bound.
    boost_db: float = Field(default=6, ge=3, le=MAX_NB_IOT_BOOST_DB, allow_inf_nan=False, description=(
        "NB-IoT power boost X in dB over the average resource element. Studio accepts 3 to 12 dB: at least 6 dB for host carriers up to 20 MHz "
        "and 3 dB for 25 MHz and wider (TS 38.104 Table 6.3.4.2-1); the upper limit of 12 dB is a Studio limit, the standard sets none."))

    @field_validator("boost_db", mode="before")
    @classmethod
    def _boost_limit_is_ours(cls, value):
        # Runs on whatever the client sent, before pydantic has judged the type: it may only ever raise ValueError. Anything that is not
        # a finite number it can compare (bool, None, containers, NaN, +-inf, an int too large for a float, a string that is not a
        # number) is left to pydantic, whose own message is then the one the client reads.
        number = _plain_number(value)
        if math.isfinite(number) and number > MAX_NB_IOT_BOOST_DB:
            raise ValueError(f"{shortest_number(number)} dB is above the largest NB-IoT boost Studio offers ({MAX_NB_IOT_BOOST_DB} dB, a Studio "
                             "limit: the standard sets no upper bound for X).")
        return value


def _plain_number(value) -> float:
    """``value`` as a float if it is a plain number or a numeric string, NaN otherwise (never raises)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return math.nan
    try:
        return float(value)
    except (ValueError, OverflowError, TypeError, ArithmeticError):     # 'x', 10**400, Decimal('sNaN'), ...
        return math.nan


def shortest_number(number: float) -> str:
    """The shortest text that reads back as ``number``, without a trailing '.0': 13 -> '13', 12.0000001 -> '12.0000001'.
    (``:g`` rounds to six digits, which printed a boost just above the cap as '12 dB is above ... 12 dB'.)"""
    text = repr(float(number))
    return text[:-2] if text.endswith(".0") else text


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
