"""Pure data for the NR FR1 standard test models (3GPP TS 38.141-1 V18.14.0 clause 4.9.2).

Nothing in this module computes a waveform. Every table names the clause it was transcribed from; numbers
were read from rendered page images of the standards, not from text extraction (formulas are garbled there).
"""
from __future__ import annotations

from dataclasses import dataclass

STANDARD = "3GPP TS 38.141-1 V18.14.0 clause 4.9.2 (NR FR1 base station test models)"
SCS_KHZ = (15, 30, 60)
MU = {15: 0, 30: 1, 60: 2}                       # TS 38.211 table 4.2-1 (FR1 subset)
MAX_CELL_ID = 1007                               # TS 38.211 N_ID^cell range
GAP_STEP_KHZ = 300                               # least common multiple of the 100 kHz raster and 15/30/60 kHz SCS

# TS 38.104 V18.15.0 table 5.3.2-1: transmission bandwidth configuration N_RB, FR1, channel bandwidth (MHz) -> N_RB.
# 3 MHz exists for 15 kHz only; the table equals the one used by the generic presets.
NRB: dict[int, dict[int, int]] = {
    15: {3: 15, 5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 35: 188, 40: 216, 45: 242, 50: 270},
    30: {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 35: 92, 40: 106, 45: 119, 50: 133,
         60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
    60: {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 35: 44, 40: 51, 45: 58, 50: 65,
         60: 79, 70: 93, 80: 107, 90: 121, 100: 135},
}

# TS 38.104 V18.15.0 table 5.3.3-1: minimum guard band (kHz) of a BS channel bandwidth, FR1.
GUARD_BAND_KHZ: dict[int, dict[int, float]] = {
    15: {3: 142.5, 5: 242.5, 10: 312.5, 15: 382.5, 20: 452.5, 25: 522.5, 30: 592.5, 35: 572.5, 40: 552.5,
         45: 712.5, 50: 692.5},
    30: {5: 505, 10: 665, 15: 645, 20: 805, 25: 785, 30: 945, 35: 925, 40: 905, 45: 1065, 50: 1045,
         60: 825, 70: 965, 80: 925, 90: 885, 100: 845},
    60: {10: 1010, 15: 990, 20: 1330, 25: 1310, 30: 1290, 35: 1630, 40: 1610, 45: 1590, 50: 1570,
         60: 1530, 70: 1490, 80: 1450, 90: 1410, 100: 1370},
}

# NB-IoT operation in NR in-band (NR-N-TM, clause 4.9.2.2.9) exists for 15 kHz only; TS 38.104 table 5.3.2-1 3...50 MHz.
NB_IOT_SCS_KHZ = 15
NB_IOT_BANDWIDTHS_MHZ = tuple(NRB[15])

MODULATION_QM = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}
QM_NAME = {qm: name for name, qm in MODULATION_QM.items()}


def valid_bandwidths(scs_khz: int) -> tuple[int, ...]:
    return tuple(NRB[scs_khz])


def n_rb(scs_khz: int, bandwidth_mhz: int) -> int:
    """N_RB of TS 38.104 table 5.3.2-1; ValueError (listing the valid choices) for a cell the table does not define."""
    try:
        return NRB[scs_khz][bandwidth_mhz]
    except KeyError:
        if scs_khz not in NRB:
            raise ValueError(f"Subcarrier spacing {scs_khz} kHz is not defined for NR FR1 test models; use 15, 30 or 60 kHz.") from None
        raise ValueError(
            f"A {bandwidth_mhz} MHz channel is not defined for {scs_khz} kHz subcarrier spacing in TS 38.104 table 5.3.2-1; "
            f"valid bandwidths: {', '.join(str(b) for b in NRB[scs_khz])} MHz.") from None


# ---------------------------------------------------------------------------------------------------------------
# TDD (TS 38.141-1 table 4.9.2.2-1, TS 38.331 TDD-UL-DL-ConfigCommon semantics, TS 38.213 clause 11.1)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TddPattern:
    scs_khz: int
    period_ms: int
    dl_slots: int
    dl_symbols: int
    ul_slots: int
    ul_symbols: int

    @property
    def slots(self) -> int:
        return self.period_ms * 2 ** MU[self.scs_khz]

    def symbol_map(self) -> str:
        """'D'/'F'/'U' per symbol of one period: the first dl_slots slots and the dl_symbols after them are downlink,
        the last ul_slots slots and the ul_symbols before them are uplink, everything else is flexible (TS 38.213 11.1)."""
        total = self.slots * 14
        symbols = ["F"] * total
        for i in range(self.dl_slots * 14 + self.dl_symbols):
            symbols[i] = "D"
        for i in range(total - self.ul_slots * 14 - self.ul_symbols, total):
            symbols[i] = "U"
        return "".join(symbols)

    def slot_maps(self) -> list[str]:
        full = self.symbol_map()
        return [full[i * 14:(i + 1) * 14] for i in range(self.slots)]

    def dl_symbols_of_slot(self, slot_in_period: int) -> int:
        """Number of leading downlink symbols of a slot (14 = full DL slot, 0 = no downlink transmission)."""
        return self.slot_maps()[slot_in_period].count("D")

    def labels(self) -> list[str]:
        out = []
        for slot in self.slot_maps():
            d, u = slot.count("D"), slot.count("U")
            if d == 14:
                out.append("D")
            elif u == 14:
                out.append("U")
            elif d or u:
                out.append("S " + "/".join(part for part in (f"{d}D" if d else "", f"{u}U" if u else "") if part))
            else:
                out.append("F")
        return out


TDD_PATTERNS: dict[int, TddPattern] = {
    15: TddPattern(15, 5, 3, 10, 1, 2),
    30: TddPattern(30, 5, 7, 6, 2, 4),
    60: TddPattern(60, 5, 14, 12, 4, 8),
}

# TS 38.141-1 table 4.9.2.2-1a (NB-IoT in-band, 15 kHz only): D S U U D with TDD-UL-DL-ConfigDedicated overrides.
# Documented for completeness; NR-N-TM is generated for FDD only (see catalog notes).
TDD_NB_IOT_SLOT_MAPS = ("DDDDDDDDDDDDDD", "DDDDDDDDDDFFUU", "UUUUUUUUUUUUUU", "UUUUUUUUUUUUUU", "DDDDDDDDDDDDDD")

STANDARD_SUBFRAMES = {"fdd": 10, "tdd": 20}      # clause 4.9.2.2: 1 radio frame FDD, 2 radio frames TDD
SUBFRAME_MS = 1


# ---------------------------------------------------------------------------------------------------------------
# PDSCH DM-RS symbol positions, TS 38.211 table 7.4.1.1.2-3: mapping type A, single-symbol DM-RS, dmrs-TypeA-Position
# pos2 (l0 = 2), dmrs-AdditionalPosition pos1 (TS 38.141-1 table 4.9.2.2-3). Key = l_d, the number of symbols from the
# first symbol of the slot to the last symbol of the scheduled PDSCH (S + L, always the last downlink symbol here).
# ---------------------------------------------------------------------------------------------------------------

DMRS_TYPE_A_POS2_ADD1: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (3, 7, (2,)),
    (8, 9, (2, 7)),
    (10, 12, (2, 9)),
    (13, 14, (2, 11)),
)


def dmrs_symbols(l_d: int) -> tuple[int, ...]:
    for low, high, positions in DMRS_TYPE_A_POS2_ADD1:
        if low <= l_d <= high:
            return positions
    raise ValueError(f"No PDSCH DM-RS position defined for a {l_d}-symbol allocation.")


# ---------------------------------------------------------------------------------------------------------------
# Test-model definitions (clauses 4.9.2.2.1 ... 4.9.2.2.9, mapping of PRBs to n_RNTI: table 4.9.2.3.2-1)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UserDef:
    rnti: int
    role: str
    qm: int                      # bits per modulation symbol
    start_symbol: int
    evm_measured: bool = False


@dataclass(frozen=True)
class ModelDef:
    id: str
    clause: str
    title: str
    description: str
    used_for: tuple[str, ...]
    users: tuple[UserDef, ...]
    structure: str                         # "uniform" | "rbg" | "single_prb"
    x_percent: int | None = None           # target share of PRBs in the boosted/de-boosted RBG set
    target_rnti: int | None = None         # user carried by the target RBGs
    target_level_db: float | None = None   # level of the target RBGs (+3, -3, -6 dB); the other user balances the power
    supports_layers: bool = False
    nb_iot: bool = False
    fdd_only: bool = False

    @property
    def suffix(self) -> str:
        """'tm3p1a' for NR-FR1-TM3.1a, 'ntm' for NR-N-TM (used in preset ids and dataset names)."""
        tail = self.id.removeprefix("NR-FR1-").removeprefix("NR-")
        return tail.lower().replace(".", "p").replace("-", "")

    def user(self, rnti: int) -> UserDef:
        return next(u for u in self.users if u.rnti == rnti)

    @property
    def highest_qm(self) -> int:
        return max(u.qm for u in self.users)


def _u(rnti, role, qm, start, evm=False):
    return UserDef(rnti, role, qm, start, evm)


_FULL = "PRB 3 to N_RB-1 (all PRBs outside the control-channel PRBs), symbols from 0"
_LOW = "PRB 0-2, symbols from 2 (PRBs below the full-band user)"
_SINGLE = "one PRB per slot at a slot-dependent position, symbols from 2"
_UF = "Unwanted emissions > "

MODELS: tuple[ModelDef, ...] = (
    ModelDef(
        "NR-FR1-TM1.1", "4.9.2.2.1", "Full-band QPSK",
        "Two QPSK users cover the whole carrier: n_RNTI 0 on PRB 3...N_RB-1 and n_RNTI 2 on PRB 0-2, all at the same level. "
        "The same structure is the model for rank-2 time alignment error tests.",
        ("BS output power", "Transmit ON/OFF power", "TAE", _UF + "Occupied bandwidth", _UF + "ACLR",
         _UF + "Operating band unwanted emissions", _UF + "Transmitter spurious emissions", "Transmitter intermodulation",
         "Receiver spurious emissions"),
        (_u(0, _FULL, 2, 0), _u(2, _LOW, 2, 2)), "uniform", supports_layers=True),
    ModelDef(
        "NR-FR1-TM1.2", "4.9.2.2.2", "Boosted and de-boosted QPSK",
        "QPSK resource-block groups (x = 40 % of the PRBs) are boosted by 3 dB, the remaining PRBs are de-boosted so that "
        "the total power equals that of N_RB-3 PRBs at 0 dB; n_RNTI 2 occupies PRB 0-2.",
        (_UF + "ACLR", _UF + "Operating band unwanted emissions"),
        (_u(0, "boosted QPSK RBGs (type 0 allocation)", 2, 0), _u(1, "de-boosted QPSK PRBs (type 0 allocation)", 2, 0),
         _u(2, _LOW, 2, 2)), "rbg", x_percent=40, target_rnti=0, target_level_db=3.0),
    ModelDef(
        "NR-FR1-TM2", "4.9.2.2.3", "Single 64QAM PRB",
        "One 64QAM PRB per slot (n_RNTI 2) walks over PRB 0, floor(N_RB/2) and N_RB-1 in consecutive slots; "
        "all other PRBs are unallocated.",
        ("Total power dynamic range (lower OFDM symbol TX power limit (OSTP) at min power) > EVM of single 64QAM PRB allocation (at min power)",
         "Total power dynamic range (lower OFDM symbol TX power limit (OSTP) at min power) > Frequency error (at min power)"),
        (_u(2, _SINGLE, 6, 2, True),), "single_prb"),
    ModelDef(
        "NR-FR1-TM2a", "4.9.2.2.4", "Single 256QAM PRB",
        "As NR-FR1-TM2 with the 64QAM PRB replaced by a 256QAM PRB.",
        ("Total power dynamic range (lower OFDM symbol TX power limit (OSTP) at min power) > EVM of single 256QAM PRB allocation (at min power)",
         "Total power dynamic range (lower OFDM symbol TX power limit (OSTP) at min power) > Frequency error (at min power)"),
        (_u(2, _SINGLE, 8, 2, True),), "single_prb"),
    ModelDef(
        "NR-FR1-TM2b", "4.9.2.2.4A", "Single 1024QAM PRB",
        "As NR-FR1-TM2 with the 64QAM PRB replaced by a 1024QAM PRB.",
        ("Total power dynamic range (lower OFDM symbol TX power limit (OSTP) at min power) > EVM of single 1024QAM PRB allocation (at min power)",
         "Total power dynamic range (lower OFDM symbol TX power limit (OSTP) at min power) > Frequency error (at min power)"),
        (_u(2, _SINGLE, 10, 2, True),), "single_prb"),
    ModelDef(
        "NR-FR1-TM3.1", "4.9.2.2.5", "Full-band 64QAM",
        "As NR-FR1-TM1.1 with both users in 64QAM (the control channel stays QPSK).",
        ("Output power dynamics > Total power dynamic range (upper OFDM symbol TX power limit (OSTP) at max power with all 64QAM PRBs allocated)",
         "Transmitted signal quality > Frequency error", "Transmitted signal quality > EVM for 64QAM modulation (at max power)"),
        (_u(0, _FULL, 6, 0, True), _u(2, _LOW, 6, 2, True)), "uniform"),
    ModelDef(
        "NR-FR1-TM3.1a", "4.9.2.2.6", "Full-band 256QAM",
        "As NR-FR1-TM1.1 with both users in 256QAM (the control channel stays QPSK).",
        ("Output power dynamics > Total power dynamic range (upper OFDM symbol TX power limit (OSTP) at max power with all 256QAM PRBs allocated)",
         "Transmitted signal quality > Frequency error", "Transmitted signal quality > EVM for 256QAM modulation (at max power)"),
        (_u(0, _FULL, 8, 0, True), _u(2, _LOW, 8, 2, True)), "uniform"),
    ModelDef(
        "NR-FR1-TM3.1b", "4.9.2.2.6A", "Full-band 1024QAM",
        "As NR-FR1-TM1.1 with both users in 1024QAM (the control channel stays QPSK).",
        ("Output power dynamics > Total power dynamic range (upper OFDM symbol TX power limit (OSTP) at max power with all 1024QAM PRBs allocated)",
         "Transmitted signal quality > Frequency error", "Transmitted signal quality > EVM for 1024QAM modulation (at max power)"),
        (_u(0, _FULL, 10, 0, True), _u(2, _LOW, 10, 2, True)), "uniform"),
    ModelDef(
        "NR-FR1-TM3.2", "4.9.2.2.7", "De-boosted 16QAM among boosted QPSK",
        "16QAM resource-block groups (x = 60 % of the PRBs, n_RNTI 1, EVM measured) are de-boosted by 3 dB; the remaining "
        "PRBs carry boosted QPSK (n_RNTI 0) to keep the total power of N_RB-3 PRBs; n_RNTI 2 occupies PRB 0-2.",
        ("Transmitted signal quality > Frequency error", "Transmitted signal quality > EVM for 16QAM modulation"),
        (_u(0, "QPSK PRBs balancing the power, EVM not measured (type 0 allocation)", 2, 0),
         _u(1, "de-boosted 16QAM RBGs, EVM measured (type 0 allocation)", 4, 0, True), _u(2, _LOW, 2, 2)),
        "rbg", x_percent=60, target_rnti=1, target_level_db=-3.0),
    ModelDef(
        "NR-FR1-TM3.3", "4.9.2.2.8", "De-boosted QPSK among boosted QPSK",
        "QPSK resource-block groups (x = 50 % of the PRBs, n_RNTI 1, EVM measured) are de-boosted by 6 dB; the remaining "
        "PRBs carry boosted QPSK (n_RNTI 0) to keep the total power of N_RB-3 PRBs; n_RNTI 2 occupies PRB 0-2.",
        ("Transmitted signal quality > Frequency error", "Transmitted signal quality > EVM for QPSK modulation"),
        (_u(0, "QPSK PRBs balancing the power, EVM not measured (type 0 allocation)", 2, 0),
         _u(1, "de-boosted QPSK RBGs, EVM measured (type 0 allocation)", 2, 0, True), _u(2, _LOW, 2, 2)),
        "rbg", x_percent=50, target_rnti=1, target_level_db=-6.0),
    ModelDef(
        "NR-N-TM", "4.9.2.2.9", "NB-IoT operation in NR in-band",
        "NR-FR1-TM1.1 on the NR carrier with one NR resource block punctured by an NB-IoT carrier (TS 36.141 N-TM) at the "
        "eligible position closest to the minimum guard band; the NB-IoT resource elements are boosted by X dB over the "
        "average resource element. FDD, 15 kHz only.",
        ("Unwanted emissions > Occupied bandwidth", "Unwanted emissions > ACLR", "Unwanted emissions > Operating band unwanted emissions",
         "Unwanted emissions > Transmitter spurious emissions", "Transmitter intermodulation", "Receiver spurious emissions",
         "NB-IoT RB power dynamic range"),
        (_u(0, _FULL, 2, 0), _u(2, _LOW, 2, 2)), "uniform", nb_iot=True, fdd_only=True),
)
MODEL_BY_ID = {m.id: m for m in MODELS}


# ---------------------------------------------------------------------------------------------------------------
# Parameters common to all models, as readable text for the catalogue (tables 4.9.2.2-2, -3, -4, clause 4.9.2.3)
# ---------------------------------------------------------------------------------------------------------------

COMMON_FIXED_PARAMETERS: tuple[tuple[str, str], ...] = (
    ("Cyclic prefix", "Normal CP (TS 38.141-1 4.9.2.2); virtual resource blocks of localized type"),
    ("Transmission bandwidth", "N_RB = maximum transmission bandwidth configuration of TS 38.104 table 5.3.2-1 (12 N_RB subcarriers)"),
    ("Duration", "1 radio frame (10 ms) for FDD, 2 radio frames (20 ms) for TDD; slots numbered 0 to 10 x 2^mu - 1 in every frame"),
    ("PDCCH (table 4.9.2.2-2)", "2 symbols from symbol 0, 1 CCE (aggregation level 1, 6 REGs = PRB 0-2 in both symbols), "
                                "QPSK, PDCCH EPRE = DM-RS EPRE, 0 dB boost, port 2000, in every DL and special slot"),
    ("PDCCH data", "108 bits per slot from the PN23 generator, scrambled with c_init = N_ID^cell (n_RNTI = 0)"),
    ("PDSCH mapping (table 4.9.2.2-3)", "Mapping type A, dmrs-TypeA-Position pos2, dmrs-AdditionalPosition 1, DM-RS configuration type 1, "
                                        "maxLength 1, PDSCH EPRE = DM-RS EPRE (0 dB)"),
    ("PDSCH DM-RS symbols", "l = 2 and 11 for 14 DL symbols; 2 and 9 for 10 or 12 DL symbols; 2 for 6 DL symbols; "
                            "even subcarriers carry DM-RS, the odd subcarriers of DM-RS symbols carry PDSCH data"),
    ("PDSCH users (table 4.9.2.2-4)", "n_RNTI 0/1 start at symbol 0, n_RNTI 2 on PRB 0-2 starts at symbol 2; PDSCH EPRE = PDCCH EPRE (0 dB)"),
    ("Antenna ports", "PDCCH 2000; PDSCH 1000 (rank 1, one layer, no precoding)"),
    ("Cell identity", "N_ID^cell = 1 for the lowest carrier, 2 for the second lowest, ...; scrambling n_ID = DM-RS N_ID = N_ID^cell, n_SCID = 0"),
    ("Data", "PN23 bits (ITU-T O.150), scrambled per TS 38.211 7.3.1.1 / 7.3.2.3, no channel coding"),
    ("Not included", "SSB, CSI-RS, PT-RS, PRACH, channel coding and CRC"),
)
