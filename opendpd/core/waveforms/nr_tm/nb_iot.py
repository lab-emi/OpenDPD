"""NB-IoT test model (N-TM) content for the NR in-band test model NR-N-TM.

TS 38.141-1 V18.14.0 clause 4.9.2.2.9 puncture one NR RB per NB-IoT carrier and fill it with the NB-IoT test model of
TS 36.141 clauses 6.1.3-6.1.6 (the only NB-IoT test model there is: "N-TM"; there is no N-TM1/2/3).  This module builds that
content at resource-element level for one 10 ms FDD frame at unit RE amplitude and leaves the OFDM modulation and the
NR host grid to ``nr_tm.grid``/``nr_tm.synth``.  Only numpy is used; everything is deterministic.

Sources
-------
* TS 36.141 V19.1.0 (2026-02) 6.1.3 (general parameters: single port, normal CP, 10 subframes, NPDCCH format 1, sync EPRE : NRS
  EPRE = 0 dB), 6.1.4 (data content: n_f = 0, n_s = 0, 'all 0' data, n_RNTI = 1000, N_ID rule, 304/200 bits per NPDCCH/NPDSCH
  subframe), 6.1.5/6.1.6 (power procedure, in-band).  The NR specification cites TS 36.141 undated; the V19.1.0 text was read
  and its change history shows no later change to 6.1.3-6.1.6 (the Rel-18 text itself was not read; resolution 8).
* TS 36.211 V19.3.0 clause 10.2: 10.2.3 NPDSCH, 10.2.4 NPBCH, 10.2.5 NPDCCH, 10.2.6 NRS, 10.2.7 NPSS/NSSS (Table 10.2.7.2.1-1 for
  b_q(m)), 10.2.8 baseband; clause 7.1.2 QPSK, 7.2 Gold sequence; 6.10.1.2 cell-specific reference signals (host CRS budget).
* TS 36.213 clause 16.4.1.4/16.6.1 (l_DataStart, bit budgets 304/200) and Table 16.8-1 (LTE in-band eligible PRBs).
* TS 36.104 V19.2.0 clause 5.7.3 (anchor carrier: M_DL in {-2,-1,0,1} -> NB-IoT carrier centre = 100 kHz raster +/- {2.5, 7.5} kHz).
* TS 38.104 Table 5.4.2.2-1 (NR RF reference frequency F_REF), Table 5.3.2-1 (N_RB) and Table 6.3.4.2-1 (NB-IoT RB power dynamic range); read in
  V18.15.0 and V18.10.0, whose copies of these tables are identical.
* TS 38.141-1 V18.14.0 4.9.2.2.9 (NR-N-TM and its power formulas), 4.9.3 (L_NB-IoT / R_NB-IoT), 3.1 (in-band definition),
  Table 4.9.2.2-1a (TDD pattern for NR with NB-IoT: defined by the standard, not generated here; Studio implements FDD only).

Validation
----------
Every signal and channel was compared with MATLAB R2026a LTE Toolbox NB-IoT functions used as a numerical oracle (symbol values to
1e-12, RE sets and mapping order, both RE budgets, every host v_shift, N_ID in all four NSSS q ranges):
``tests/data/nr_tm/nbiot_oracle.json`` holds a trimmed dump for 13 (N_ID, host cell id, operation mode) combinations and
``tests/unit/test_nr_tm_nbiot.py`` reproduces it.  The only values compared at a looser 2e-11 are the NSSS ones: MATLAB's own
output deviates by up to 8e-12 from the exact definition (argument reduction of exp(-j pi u n'(n'+1)/131)), the phases here are
integer multiples of 1/524 turn and exact to 1e-15 (checked against a 40-digit evaluation).  No MathWorks code is used or copied.

Conventions and resolutions of under-specified points (numbering follows the research report ``nbiot_ntm.md`` section 7)
------------------------------------------------------------------------------------------------------------------------
1. REs of the punctured RB that carry no NB-IoT signal (the control symbols l = 0..2, the host CRS positions and the muted
   NRS port-2001 positions) keep the NR-FR1-TM1.1 content, scaled by ``nr_gain``.  38.141-1 only says "puncturing one NR RB" but
   distinguishes the N_NB-IoT REs "containing NB-IoT signal" from the N_NR NR REs; TS 36.141 6.1.6 says the NB-IoT PRB "also contains
   certain REs for the hosting carrier".  ``mask`` marks exactly the NB-IoT-signal REs; the caller keeps NR everywhere else.
2. 36.141 prescribes 304 (stand-alone/guard band) or 200 (in-band) bits per NPDCCH/NPDSCH subframe although it also says "single
   antenna port".  Those numbers are the RE budgets of a 2-port NRS and, in-band, of 4 host CRS ports with a 3-symbol control
   region (MATLAB reproduces them with NBRefP = 2, CellRefP = 4, ControlRegionSize = 3).  Resolution: the port-2001 NRS RE
   positions are reserved (muted) and the signal is transmitted on port 2000 only (``nrs_ports=2``, default); the in-band host
   budget is 4 CRS ports, l_DataStart = 3 (``host_crs_ports=4``, ``control_symbols=3``).  This gives 100 data REs (200 bits)
   per in-band subframe and 1085 NB-IoT-signal REs per frame for host v_shift in {0, 1, 3, 4}.  The NPSS has no subcarrier 11,
   so a host CRS RE on k = 11 (host v_shift 2 or 5) is not part of it and the NPSS keeps 109 instead of 105 REs: 1089 per frame.
   The "natural" one-port budget (160/108 REs, 320/216 bits) is selectable with ``nrs_ports=1`` and verified as well.
3. NPDSCH codeword: every subframe is its own codeword (N_SF = 1, n_f = 0) with c_init = 1000 * 2^14 + floor(n_s/2) * 2^9 + N_ID,
   n_s = first slot of the subframe (36.141 is silent on N_SF and bundling; it changes only the PRBS phase, not the statistics).
   NPDCCH (subframe 1): c_init = floor(n_s/2) * 2^9 + N_ID.  NPBCH: first 200 bits of the 1600-bit block, n_f = 0.
4. N_ID^Ncell = 97 + 6 n + max(0, m - 1) (36.141 6.1.4).  The worked values of the standard (103 for the lowest host, 110 for the
   2nd lowest) fix n as the rank (1-based, ascending frequency) of the host carrier among the NB-IoT hosts and m as the host
   cell ID.  Both NB-IoT carriers (lower and upper) of one host share the N_ID ("NB-IoT carrier(s) within the lowest E-UTRA
   carrier").  The rule reaches 503 (the largest NB-IoT N_ID) only for hosts up to cell ID 407 - 6 n; larger ones raise ValueError.
   N_ID mod 6 equals the host's v_shift for host cell IDs >= 1, which is why the host CRS positions equal those of N_ID.
   Host cell ID 0 (allowed for NR carriers) is the one case where they differ: see 9.
5. Power counting window: whole-frame energy balance (``power_split``): N_NB-IoT = REs carrying NB-IoT signal in the 10 ms frame
   (of all NB-IoT carriers of the NR carrier), N_NR = N_RB * 12 * 140 - N_NB-IoT, P_avg = P_rated / (N_RB * 12) per RE, every RE
   of the NR grid loaded at unit amplitude.  Balancing every OFDM symbol on its own instead would move the NR level of the symbols
   that carry NB-IoT by up to -0.41 dB (3 MHz), -0.10 dB (10 MHz), -0.02 dB (50 MHz) and leave the other symbols at P_avg, i.e. up
   to +0.63 / +0.17 / +0.03 dB above the frame-balance level (X = 6 dB, one carrier; checked in the tests).
6. The eligible-RB comb assumes that the NR RF reference frequency F_REF (centre of RE k = 0 of PRB N_RB/2 for even N_RB, of RE
   k = 6 of PRB floor(N_RB/2) for odd N_RB) is a multiple of 100 kHz, as for the 100 kHz-raster NB-IoT bands.  A different
   ``f_ref_mod_khz`` (F_REF mod 100 kHz) shifts the comb; the standard defines no table of eligible NR RB indices, the rule is
   derived from the TS 36.104 5.7.3 anchor raster exactly as TS 36.213 Table 16.8-1 follows from it for LTE (verified in tests).
   L_NB-IoT / R_NB-IoT (4.9.3) are the lowest / highest eligible RB: "the eligible in-band position closest to the NR minimum
   guard band".
7. NPDSCH 16QAM (an optional variant of 36.141 used only in its own EVM test, never in an NR test) is not offered: QPSK only.
8. The N-TM is one 10 ms frame (n_f = 0) that is looped: the NSSS (present in even frames only in a real network) is sent in
   subframe 9 of every looped frame with theta_f = 0.
9. When the host v_shift differs from N_ID mod 6 (host cell ID 0, or a hand-picked N_ID) the standards do not say which CRS
   positions the NPDCCH/NPDSCH budget avoids.  The MATLAB oracle (13 combinations, verified) uses the v_shift of N_ID^Ncell for
   NPBCH/NPDCCH/NPDSCH and the host's v_shift for the NPSS/NSSS skip; that is what ``nb_iot_frame`` does.  For every host cell
   ID >= 1 under the 36.141 N_ID rule both are the same.

Further fixed choices: FDD and 15 kHz subcarrier spacing only. The standards describe NB-IoT in-band operation for 15 kHz hosts only
(an NB-IoT carrier of 180 kHz is one NR resource block only at 15 kHz; no text for 30 or 60 kHz hosts was found). TDD is defined as well
(TS 38.141-1 Table 4.9.2.2-1a: D S U U D at 15 kHz, 5 ms; TS 36.141 6.1.3: NB-IoT TDD N-TM with UL/DL configuration 1, special subframe
configuration 7 and two frames), but it is not implemented here: the NB-IoT content of the special and uplink subframes of that
combination was not built and no oracle exists for it. The NB-IoT RE grid is mapped 1:1 onto
the NR resource grid (an NB-IoT PRB is an NR RB at 15 kHz with the same CP structure; the per-symbol common phase of NB-IoT 10.2.8
versus NR 38.211 5.3.1 differs by a factor common to the whole PRB and does not change PSD/ACLR); NPSS/NSSS REs that overlap
host CRS REs are skipped but counted in the sequence mapping (10.2.7.1.2, 10.2.7.2.2); the NPBCH assumes 4 CRS ports and the NRS
ports 2000/2001 irrespective of the configuration (10.2.4.4, v_shift from N_ID^Ncell).

Array layout: ``values[subframe, symbol, subcarrier]`` = ``[10, 14, 12]`` complex, ``mask`` the same shape (bool).  NR slot s of
a 15 kHz FDD carrier is NB-IoT subframe ``s % 10``.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = [
    "ANCHOR_OFFSETS_KHZ", "MAX_N_ID", "NbIotCarrierPlan", "NbIotOverlayPlan",
    "eligible_rbs", "frame_re_count", "lower_rb", "upper_rb", "max_boost_db", "nb_iot_frame", "nb_iot_n_id", "nb_iot_n_ids",
    "nb_iot_overlay_plan", "power_dynamic_range_floor_db", "power_split",
]

N_SC = 12                 # subcarriers per RB
N_SYMB = 14               # OFDM symbols per subframe (normal CP)
N_SUBFRAMES = 10          # subframes per frame
MAX_N_ID = 503            # largest N_ID^Ncell of NB-IoT
MIN_NRB, MAX_NRB = 15, 275
_NC = 1600                # Gold sequence offset (36.211 7.2)
_NRS_MAX_DL_RB = 110      # N_RB^max,DL of 36.211 10.2.6.1
_NRS_PORTS = (2000, 2001)
# TS 36.104 5.7.3 NOTE 3: NPSS/NSSS carriers use M_DL in {-2,-1,0,1} -> offsets 0.0025 (2 M_DL + 1) MHz from the 100 kHz raster.
ANCHOR_OFFSETS_KHZ = (Fraction(-15, 2), Fraction(-5, 2), Fraction(5, 2), Fraction(15, 2))
# NSSS Table 10.2.7.2.1-1: b_q(m), m = 0..127, q = 0..3 ('+' = +1, '-' = -1); rows 0, 31, 63, 127 of the order-128 Hadamard matrix.
_B_Q = tuple(
    "".join(rows) for rows in (
        ("++++++++++++++++++++++++++++++++",) * 4,
        ("+--+-++--++-+--+-++-+--++--+-++-",) * 4,
        ("+--+-++--++-+--+-++-+--++--+-++-", "-++-+--++--+-++-+--+-++--++-+--+",
         "+--+-++--++-+--+-++-+--++--+-++-", "-++-+--++--+-++-+--+-++--++-+--+"),
        ("+--+-++--++-+--+-++-+--++--+-++-", "-++-+--++--+-++-+--+-++--++-+--+",
         "-++-+--++--+-++-+--+-++--++-+--+", "+--+-++--++-+--+-++-+--++--+-++-"),
    ))
_S_NPSS = (1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1)  # S(l), l = 3..13 (10.2.7.1.1)


# ---------------------------------------------------------------------------------------------------------------------
# eligible RBs, N_ID, power
# ---------------------------------------------------------------------------------------------------------------------
def _nrb(nrb) -> int:
    if isinstance(nrb, bool) or not isinstance(nrb, (int, np.integer)) or not MIN_NRB <= int(nrb) <= MAX_NRB:
        raise ValueError(f"nrb must be an integer in {MIN_NRB}..{MAX_NRB} (15 kHz NR carrier hosting NB-IoT), got {nrb!r}")
    return int(nrb)


def raster_offset_khz(centre_khz) -> Fraction:
    """Signed distance in kHz (-50, 50] of ``centre_khz`` to the nearest multiple of the 100 kHz channel raster."""
    r = Fraction(centre_khz) % 100
    return r - 100 if r > 50 else r


def _prb_centre_khz(rb: int, nrb: int) -> Fraction:
    """Centre of NR PRB ``rb`` (15 kHz) relative to F_REF (38.104 Table 5.4.2.2-1)."""
    if nrb % 2 == 0:
        return 180 * (rb - nrb // 2) + Fraction(165, 2)      # F_REF = centre of RE k = 0 of PRB N/2
    return 180 * (rb - nrb // 2) + Fraction(-15, 2)          # F_REF = centre of RE k = 6 of PRB floor(N/2)


@functools.lru_cache(maxsize=256)
def _eligible(nrb: int, f0: Fraction) -> tuple[int, ...]:
    return tuple(rb for rb in range(nrb) if raster_offset_khz(f0 + _prb_centre_khz(rb, nrb)) in ANCHOR_OFFSETS_KHZ)


def eligible_rbs(nrb: int, f_ref_mod_khz=0) -> list[int]:
    """NR RB indices (0-based, ascending) whose PRB centre is a valid anchor-carrier position (resolution 6)."""
    nrb = _nrb(nrb)
    rbs = _eligible(nrb, Fraction(f_ref_mod_khz).limit_denominator(1000))
    if not rbs:
        raise ValueError(f"no NB-IoT in-band position is eligible for N_RB={nrb} with F_REF mod 100 kHz = {f_ref_mod_khz}")
    return list(rbs)


def lower_rb(nrb: int, f_ref_mod_khz=0) -> int:
    """L_NB-IoT: the lowest eligible RB (closest to the lower NR minimum guard band)."""
    return eligible_rbs(nrb, f_ref_mod_khz)[0]


def upper_rb(nrb: int, f_ref_mod_khz=0) -> int:
    """R_NB-IoT: the highest eligible RB (closest to the upper NR minimum guard band)."""
    return eligible_rbs(nrb, f_ref_mod_khz)[-1]


def nb_iot_n_id(host_cell_id: int, ordinal: int) -> int:
    """N_ID^Ncell = 97 + 6 n + max(0, m - 1) of TS 36.141 6.1.4 (n = ``ordinal`` >= 1, m = ``host_cell_id``; 0 = stand-alone)."""
    if isinstance(host_cell_id, bool) or not isinstance(host_cell_id, (int, np.integer)) or int(host_cell_id) < 0:
        raise ValueError(f"host cell ID must be an integer >= 0, got {host_cell_id!r}")
    if isinstance(ordinal, bool) or not isinstance(ordinal, (int, np.integer)) or int(ordinal) < 1:
        raise ValueError(f"NB-IoT carrier ordinal must be an integer >= 1, got {ordinal!r}")
    n_id = 97 + 6 * int(ordinal) + max(0, int(host_cell_id) - 1)
    if n_id > MAX_N_ID:
        raise ValueError(
            f"NB-IoT N_ID {n_id} (host cell ID {host_cell_id}, NB-IoT host #{ordinal}) exceeds {MAX_N_ID}; "
            f"use a host cell ID of at most {MAX_N_ID - 96 - 6 * int(ordinal)}")
    return n_id


def nb_iot_n_ids(host_cell_ids) -> list[int]:
    """N_ID of the NB-IoT carriers inside each host, hosts listed in ascending frequency order (ordinal = index + 1)."""
    return [nb_iot_n_id(m, i + 1) for i, m in enumerate(host_cell_ids)]


def _frame_re_count(nrb: int) -> int:
    return nrb * N_SC * N_SYMB * N_SUBFRAMES


def max_boost_db(nrb: int, n_nb_re: int) -> float:
    """Exclusive upper limit of X (dB): at 10 log10(N_tot / N_NB-IoT) all power sits on the NB-IoT REs and P_NR reaches 0."""
    n_tot = _frame_re_count(_nrb(nrb))
    if not 0 < n_nb_re < n_tot:
        raise ValueError(f"n_nb_re must be in 1..{n_tot - 1}, got {n_nb_re}")
    return 10 * math.log10(n_tot / n_nb_re)


def power_split(nrb: int, n_nb_re: int, boost_db: float) -> tuple[float, float]:
    """Amplitude gains ``(nr_gain, nb_gain)`` for unit-amplitude NR and NB-IoT REs (TS 38.141-1 4.9.2.2.9, resolution 5).

    With P_avg = 1 per RE: P_NB-IoT = 10^(X/10) and P_NR = (N_tot - P_NB-IoT * N_NB) / (N_tot - N_NB), N_tot = N_RB * 12 * 140
    REs per frame and ``n_nb_re`` = number of NB-IoT-signal REs of the frame (all NB-IoT carriers of the carrier).  The frame
    energy is conserved: nr_gain^2 (N_tot - N_NB) + nb_gain^2 N_NB = N_tot.
    """
    n_tot = _frame_re_count(_nrb(nrb))
    if isinstance(n_nb_re, bool) or not isinstance(n_nb_re, (int, np.integer)) or not 0 < int(n_nb_re) < n_tot:
        raise ValueError(f"n_nb_re must be an integer in 1..{n_tot - 1}, got {n_nb_re!r}")
    if not isinstance(boost_db, (int, float, np.integer, np.floating)) or not math.isfinite(boost_db):
        raise ValueError(f"boost_db must be a finite number, got {boost_db!r}")
    n_nb = int(n_nb_re)
    g = 10.0 ** (float(boost_db) / 10.0)
    nr_power = (n_tot - n_nb * g) / (n_tot - n_nb)
    if nr_power <= 0:
        raise ValueError(
            f"NB-IoT boost of {boost_db:g} dB on {n_nb} REs leaves no power for the NR REs of a {nrb}-RB carrier "
            f"(X must stay below {max_boost_db(nrb, n_nb):.2f} dB)")
    return math.sqrt(nr_power), math.sqrt(g)


def power_dynamic_range_floor_db(bandwidth_mhz: int, nrb: int, rb: int) -> int:
    """Smallest NB-IoT RB power dynamic range X (dB) that TS 38.104 Table 6.3.4.2-1 allows a BS to declare (D.43) for an NB-IoT RB.

    +6 dB for 3, 5 and 10 MHz channels at any position and for 15 / 20 MHz channels inside the central 77 / 102 resource blocks (plus 15 kHz at
    each edge); +6 dB also for 25 MHz and wider inside the central 90 % of the channel bandwidth; +3 dB at the other positions. ``rb`` is the
    punctured NR RB (15 kHz) of a host with ``nrb`` resource blocks; the whole RB must lie inside the region for the +6 dB row to apply.
    """
    nrb = _nrb(nrb)
    if not 0 <= int(rb) < nrb:
        raise ValueError(f"rb must be in 0..{nrb - 1}, got {rb!r}")
    if bandwidth_mhz <= 10:
        return 6
    half_khz = {15: Fraction(77 * 180 + 30, 2), 20: Fraction(102 * 180 + 30, 2)}.get(bandwidth_mhz, Fraction(450 * bandwidth_mhz))     # +-45 % of the channel bandwidth, in kHz
    centre = _prb_centre_khz(int(rb), nrb)
    return 6 if centre - 90 >= -half_khz and centre + 90 <= half_khz else 3


# ---------------------------------------------------------------------------------------------------------------------
# sequences (TS 36.211 7.1.2, 7.2, 10.2.6, 10.2.7)
# ---------------------------------------------------------------------------------------------------------------------
def gold_sequence(c_init: int, n: int) -> np.ndarray:
    """Length-31 Gold sequence c(0..n-1) of TS 36.211 7.2 (uint8)."""
    if not 0 <= c_init < 2 ** 31:
        raise ValueError(f"c_init out of range: {c_init}")
    total = _NC + n + 31
    x1 = [0] * total
    x2 = [0] * total
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for i in range(total - 31):
        x1[i + 31] = x1[i + 3] ^ x1[i]
        x2[i + 31] = x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]
    return (np.array(x1[_NC:_NC + n], dtype=np.uint8) ^ np.array(x2[_NC:_NC + n], dtype=np.uint8))


def qpsk(bits) -> np.ndarray:
    """TS 36.211 7.1.2 QPSK: b(2i) -> I, b(2i+1) -> Q, unit-power constellation points (+-1 +- j) / sqrt(2)."""
    b = np.asarray(bits, dtype=np.int64)
    return ((1 - 2 * b[0::2]) + 1j * (1 - 2 * b[1::2])) / math.sqrt(2)


def _theta_f(c: np.ndarray) -> np.ndarray:
    """NPBCH phase theta_f(i) in {1, -1, j, -j} for (c(2i), c(2i+1)) = 00, 01, 10, 11 (10.2.4.4)."""
    a, b = c[0::2], c[1::2]
    return np.where(a == 0, np.where(b == 0, 1, -1), np.where(b == 0, 1j, -1j))


def crs_positions(v_shift: int, ports: int = 4) -> set:
    """(k, l) of the cell-specific reference signals of antenna ports 0..ports-1 in one PRB and subframe, normal CP (6.10.1.2)."""
    s = set()
    for slot in (0, 1):
        for p in range(ports):
            if p < 2:
                for l in (0, 4):
                    v = (0 if p == 0 else 3) if l == 0 else (3 if p == 0 else 0)
                    s.update(((6 * m + (v + v_shift) % 6), l + 7 * slot) for m in (0, 1))
            else:
                v = 3 * slot if p == 2 else 3 + 3 * slot
                s.update(((6 * m + (v + v_shift) % 6), 1 + 7 * slot) for m in (0, 1))
    return s


def nrs_positions(v_shift: int, port: int) -> list:
    """(k, l, slot, m) of the NRS of ``port`` (2000/2001) in one subframe: l = 5, 6 of each slot (10.2.6.2)."""
    if port not in _NRS_PORTS:
        raise ValueError(f"NRS port must be 2000 or 2001, got {port}")
    out = []
    for slot in (0, 1):
        for l in (5, 6):
            v = (0 if l == 5 else 3) if port == 2000 else (3 if l == 5 else 0)
            out.extend((6 * m + (v + v_shift) % 6, l + 7 * slot, slot, m) for m in (0, 1))
    return out


def nrs_values(n_id: int, port: int, subframe: int) -> dict:
    """{(k, l): r} of the NRS of ``port`` in ``subframe`` (10.2.6.1: c_init = 2^10 (7 (n_s + 1) + l + 1)(2 N_ID + 1) + 2 N_ID + 1)."""
    out = {}
    for k, l, slot, m in nrs_positions(n_id % 6, port):
        c_init = (1 << 10) * (7 * (2 * subframe + slot + 1) + (l - 7 * slot) + 1) * (2 * n_id + 1) + 2 * n_id + 1
        c = gold_sequence(c_init, 2 * (_NRS_MAX_DL_RB + 1))
        mp = m + _NRS_MAX_DL_RB - 1
        out[(k, l)] = ((1 - 2 * int(c[2 * mp])) + 1j * (1 - 2 * int(c[2 * mp + 1]))) / math.sqrt(2)
    return out


def _phasor(numerator: np.ndarray, turns: int) -> np.ndarray:
    """exp(-j 2 pi numerator / turns) with the integer numerator reduced first (no argument-reduction error)."""
    return np.exp(-2j * np.pi * (np.asarray(numerator, dtype=np.int64) % turns) / turns)


def npss_symbols() -> np.ndarray:
    """NPSS d_l(n), indexed ``[l - 3, n]`` for l = 3..13, n = 0..10 (10.2.7.1.1 with u = 5; n (n + 1) / 2 is an integer)."""
    n = np.arange(11)
    zc = _phasor(5 * (n * (n + 1) // 2), 11)
    return np.asarray(_S_NPSS, dtype=float)[:, None] * zc[None, :]


def nsss_sequence(n_id: int, n_f: int = 0) -> np.ndarray:
    """NSSS d(0..131) of 10.2.7.2.1 for radio frame ``n_f`` (even)."""
    if not 0 <= n_id <= MAX_N_ID:
        raise ValueError(f"N_ID must be 0..{MAX_N_ID}, got {n_id}")
    n = np.arange(132)
    n1 = n % 131
    u = n_id % 126 + 3
    b = np.array([1.0 if ch == "+" else -1.0 for ch in _B_Q[n_id // 126]])
    # theta_f = 33/132 ((n_f/2) mod 4) = a/4 turns, so every phase is an integer multiple of 1/524 turn: exp(-j 2 pi theta_f n)
    # exp(-j pi u n'(n'+1)/131) = exp(-j 2 pi (131 (a n mod 4) + 4 (u n'(n'+1)/2 mod 131)) / 524)
    a = (n_f // 2) % 4
    return b[n % 128] * _phasor(131 * ((a * n) % 4) + 4 * ((u * (n1 * (n1 + 1) // 2)) % 131), 524)


def data_res(l_start: int, excluded) -> list:
    """RE list of an NPDCCH/NPDSCH subframe: increasing k first, then l (10.2.3.4, 10.2.5.5)."""
    return [(k, l) for l in range(l_start, N_SYMB) for k in range(N_SC) if (k, l) not in excluded]


# ---------------------------------------------------------------------------------------------------------------------
# one frame
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _Layout:
    """Where the NB-IoT signal sits in one frame. It depends on the N_ID only through N_ID mod 6 (the NRS and CRS shift), so it is
    shared by all NB-IoT cell identities; the values (Gold sequences) are what depends on the full N_ID."""
    nrs: tuple             # (k, l, slot, m) of the NRS of port 2000
    data: tuple            # (k, l) of an NPDCCH / NPDSCH subframe
    npss: tuple            # (k, l) of subframe 5
    nsss_index: tuple      # indices into the 132 NSSS symbols that are transmitted (the others overlap host CRS)
    nsss: tuple            # (k, l) of subframe 9
    npbch: tuple           # (k, l) of subframe 0
    mask: np.ndarray       # [10, 14, 12] bool, read-only: True on the REs that carry NB-IoT signal
    n_re: int              # mask.sum(), NB-IoT-signal REs per frame


@functools.lru_cache(maxsize=256)
def _layout(v_nid: int, inband: bool, host_v_shift: int, nrs_ports: int, host_crs_ports: int, control_symbols: int) -> _Layout:
    sync_crs = crs_positions(host_v_shift, host_crs_ports) if inband else set()    # NPSS/NSSS skip the host CRS
    reserved = set()
    for p in _NRS_PORTS[:nrs_ports]:
        reserved.update((k, l) for k, l, _, _ in nrs_positions(v_nid, p))
    if inband:
        reserved |= crs_positions(v_nid, host_crs_ports)                          # data budget: CRS of N_ID's v_shift
    data = tuple(data_res(control_symbols if inband else 0, reserved))
    nrs = tuple(nrs_positions(v_nid, 2000))
    # NPSS, subframe 5: k = 0..10, l = 3..13; elements overlapping host CRS are not transmitted (10.2.7.1.2)
    npss = tuple((k, l) for l in range(3, N_SYMB) for k in range(N_SC - 1) if (k, l) not in sync_crs)
    # NSSS, subframe 9 (n_f = 0 even): d(n) fills k = 0..11 first, then l = 3..13; CRS overlaps are counted but not used
    full = [(k, l) for l in range(3, N_SYMB) for k in range(N_SC)]
    nsss_index = tuple(i for i, kl in enumerate(full) if kl not in sync_crs)
    nsss = tuple(full[i] for i in nsss_index)
    # NPBCH, subframe 0: REs l >= 3 free of CRS (ports 0-3) and NRS (ports 2000/2001), v_shift from N_ID^Ncell irrespective of the configuration (10.2.4.4)
    npbch_excl = crs_positions(v_nid, 4) | {(k, l) for p in _NRS_PORTS for k, l, _, _ in nrs_positions(v_nid, p)}
    npbch = tuple(data_res(3, npbch_excl))
    mask = np.zeros((N_SUBFRAMES, N_SYMB, N_SC), dtype=bool)
    for sf in range(N_SUBFRAMES):
        if sf not in (5, 9):                                     # no NRS in the NPSS/NSSS subframes
            for k, l, _, _ in nrs:
                mask[sf, l, k] = True
    for sf, res in ((0, npbch), (1, data), (2, data), (3, data), (4, data), (5, npss), (6, data), (7, data), (8, data), (9, nsss)):
        for k, l in res:
            mask[sf, l, k] = True
    mask.flags.writeable = False
    return _Layout(nrs, data, npss, nsss_index, nsss, npbch, mask, int(mask.sum()))


@functools.lru_cache(maxsize=64)
def _build_frame(n_id: int, inband: bool, host_v_shift: int, nrs_ports: int, host_crs_ports: int, control_symbols: int):
    lay = _layout(n_id % 6, inband, host_v_shift, nrs_ports, host_crs_ports, control_symbols)
    values = np.zeros((N_SUBFRAMES, N_SYMB, N_SC), dtype=np.complex128)

    def put(sf, res, vals):
        for (k, l), v in zip(res, vals):
            values[sf, l, k] = v

    for sf in range(N_SUBFRAMES):
        if sf not in (5, 9):
            nrs = nrs_values(n_id, 2000, sf)
            put(sf, list(nrs), list(nrs.values()))
    d = npss_symbols()
    put(5, lay.npss, [d[l - 3, k] for k, l in lay.npss])
    put(9, lay.nsss, nsss_sequence(n_id, 0)[list(lay.nsss_index)])
    # NPBCH, subframe 0: 200 'all 0' bits scrambled with c_init = N_ID, then theta_f (n_f = 0)
    c_f = gold_sequence((n_id + 1) * 1 * 2 ** 9 + n_id, 200)
    put(0, lay.npbch, qpsk(gold_sequence(n_id, 200)) * _theta_f(c_f))
    # NPDCCH (subframe 1, format 1: NCCE 0 + 1 = 12 subcarriers) and NPDSCH (SF 2-4, 6-8, n_RNTI = 1000), 'all 0' data
    put(1, lay.data, qpsk(gold_sequence(1 * 2 ** 9 + n_id, 2 * len(lay.data))))
    for sf in (2, 3, 4, 6, 7, 8):
        put(sf, lay.data, qpsk(gold_sequence(1000 * 2 ** 14 + sf * 2 ** 9 + n_id, 2 * len(lay.data))))
    values.flags.writeable = False
    return values, lay.mask


def frame_re_count(n_id: int, *, mode: str = "inband", host_cell_id: int | None = None, nrs_ports: int = 2,
                   host_crs_ports: int = 4, control_symbols: int = 3) -> int:
    """Number of REs of one frame that carry NB-IoT signal (``nb_iot_frame(...)[1].sum()``), without computing any sequence.

    The positions depend on the N_ID only through N_ID mod 6 and on the host cell ID mod 6 (resolution 9), so this is a lookup in
    a layout shared by all cell identities; plan validation uses it to check the power budget in closed form.
    """
    n_id, *rest = _frame_key(n_id, mode, host_cell_id, nrs_ports, host_crs_ports, control_symbols)
    return _layout(n_id % 6, *rest).n_re


def _frame_key(n_id, mode, host_cell_id, nrs_ports, host_crs_ports, control_symbols) -> tuple:
    """Check the arguments of ``nb_iot_frame`` / ``frame_re_count`` and return the key of ``_build_frame``."""
    if isinstance(n_id, bool) or not isinstance(n_id, (int, np.integer)) or not 0 <= int(n_id) <= MAX_N_ID:
        raise ValueError(f"N_ID must be an integer in 0..{MAX_N_ID}, got {n_id!r}")
    if mode not in ("inband", "standalone"):
        raise ValueError(f"mode must be 'inband' or 'standalone', got {mode!r}")
    if nrs_ports not in (1, 2):
        raise ValueError(f"nrs_ports must be 1 or 2, got {nrs_ports!r}")
    if host_crs_ports not in (1, 2, 4):
        raise ValueError(f"host_crs_ports must be 1, 2 or 4, got {host_crs_ports!r}")
    if control_symbols not in (0, 1, 2, 3, 4):
        raise ValueError(f"control_symbols must be 0..4, got {control_symbols!r}")
    host = int(n_id) if host_cell_id is None else host_cell_id
    if isinstance(host, bool) or not isinstance(host, (int, np.integer)) or int(host) < 0:
        raise ValueError(f"host cell ID must be an integer >= 0, got {host_cell_id!r}")
    return int(n_id), mode == "inband", int(host) % 6, nrs_ports, host_crs_ports, control_symbols


def nb_iot_frame(n_id: int, *, mode: str = "inband", host_cell_id: int | None = None, nrs_ports: int = 2,
                 host_crs_ports: int = 4, control_symbols: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """One 10 ms FDD frame of the NB-IoT test model at unit RE amplitude: ``(values, mask)``, both ``[10, 14, 12]``.

    Subframe map (n_f = 0): 0 NPBCH + NRS, 1 NPDCCH + NRS, 2-4 and 6-8 NPDSCH (n_RNTI 1000, QPSK) + NRS, 5 NPSS, 9 NSSS.
    ``mask`` is True on the REs that carry NB-IoT signal (NRS on port 2000 only, muted port-2001 positions are False).
    ``mode="inband"`` (default) uses the 36.141 in-band budget: ``host_crs_ports`` = 4 CRS ports (the data channels avoid the
    CRS REs of v_shift = N_ID mod 6, the NPSS/NSSS skip those of v_shift = ``host_cell_id`` mod 6, default ``n_id``: equal under
    the 36.141 N_ID rule, resolution 9) and ``control_symbols`` = 3; ``"standalone"`` (guard band budget) starts at l = 0 and has
    no CRS.  ``nrs_ports=2`` reserves the port-2001 NRS positions (304/200-bit budgets, resolution 2); ``1`` is the natural
    one-port budget.  The arrays are copies and may be modified.
    """
    values, mask = _build_frame(*_frame_key(n_id, mode, host_cell_id, nrs_ports, host_crs_ports, control_symbols))
    return values.copy(), mask.copy()


# ---------------------------------------------------------------------------------------------------------------------
# overlay plan for the NR host grid (consumed by nr_tm.grid / synth)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class NbIotCarrierPlan:
    """One NB-IoT carrier: the punctured NR RB, its N_ID and the unit-amplitude frame content ``[10, 14, 12]``."""
    side: str            # "lower" or "upper"
    rb: int              # 0-based NR RB (CRB of the carrier grid) that is punctured
    n_id: int
    values: np.ndarray   # complex, zero outside ``mask``; read-only
    mask: np.ndarray     # bool, True on the REs carrying NB-IoT signal; read-only

    @property
    def n_re(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True, eq=False)
class NbIotOverlayPlan:
    """How to overlay the NB-IoT carriers of one NR host carrier (15 kHz, FDD) on its resource grid.

    Per NR slot ``s`` (= NB-IoT subframe ``s % 10``) every NR RE is scaled by ``nr_gain``; on the RB ``rb`` of each carrier the REs
    where ``mask[s % 10]`` is True are replaced by ``nb_gain * values[s % 10]`` (transposed to ``[subcarrier, symbol]``).
    """
    nrb: int
    boost_db: float
    host_cell_id: int
    f_ref_mod_khz: Fraction
    carriers: tuple
    n_nb_re: int         # NB-IoT-signal REs per 10 ms frame, all carriers of this host
    nr_gain: float
    nb_gain: float

    @property
    def rbs(self) -> list[int]:
        return [c.rb for c in self.carriers]

    def overlay_slot(self, grid, slot_in_frame: int) -> np.ndarray:
        """Return the NR slot grid ``[12 * nrb, 14]`` (unit-amplitude NR content) with the NB-IoT overlay applied (new array)."""
        g = np.asarray(grid)
        if g.shape != (N_SC * self.nrb, N_SYMB):
            raise ValueError(f"slot grid must have shape {(N_SC * self.nrb, N_SYMB)}, got {g.shape}")
        out = g.astype(np.complex128) * self.nr_gain
        sf = int(slot_in_frame) % N_SUBFRAMES
        for car in self.carriers:
            sel = car.mask[sf].T                                   # [subcarrier, symbol]
            block = out[N_SC * car.rb:N_SC * (car.rb + 1)]         # view
            block[sel] = self.nb_gain * car.values[sf].T[sel]
        return out

    def overlay(self, grid) -> np.ndarray:
        """Apply the overlay to a whole grid ``[12 * nrb, 14 * n_slots]``; slot s uses NB-IoT subframe ``s % 10`` (new array)."""
        g = np.asarray(grid)
        if g.ndim != 2 or g.shape[0] != N_SC * self.nrb or g.shape[1] % N_SYMB:
            raise ValueError(f"grid must have shape [{N_SC * self.nrb}, 14 * n_slots], got {g.shape}")
        out = np.empty(g.shape, dtype=np.complex128)
        for s in range(g.shape[1] // N_SYMB):
            out[:, N_SYMB * s:N_SYMB * (s + 1)] = self.overlay_slot(g[:, N_SYMB * s:N_SYMB * (s + 1)], s)
        return out


def nb_iot_overlay_plan(nrb: int, position: str, boost_db: float, host_cell_ids, *, host_index: int = 0, rbs=None,
                        f_ref_mod_khz=0) -> NbIotOverlayPlan:
    """Plan the NB-IoT overlay of NR host carrier ``host_index`` (``nrb`` = its N_RB, 15 kHz).

    ``host_cell_ids``: cell IDs of all NR carriers that host NB-IoT, ascending frequency (a bare int = one host); the host's rank
    gives the N_ID ordinal (resolution 4).  ``position``: "lower" (L_NB-IoT), "upper" (R_NB-IoT) or "both"; both carriers of the
    host share the N_ID.  ``rbs`` overrides the punctured RBs (each must be in ``eligible_rbs``; side names then follow
    the order lower/upper by RB).  The gains come from ``power_split`` with N_NB-IoT summed over the carriers.
    """
    nrb = _nrb(nrb)
    if position not in ("lower", "upper", "both"):
        raise ValueError(f"position must be 'lower', 'upper' or 'both', got {position!r}")
    def is_int(v):
        return isinstance(v, (int, np.integer)) and not isinstance(v, bool)
    if is_int(host_cell_ids):
        hosts = [int(host_cell_ids)]
    elif isinstance(host_cell_ids, (list, tuple, np.ndarray)) and all(is_int(m) for m in host_cell_ids):
        hosts = [int(m) for m in host_cell_ids]
    else:
        raise ValueError(f"host_cell_ids must be an int or a sequence of ints, got {host_cell_ids!r}")
    if not hosts or not 0 <= host_index < len(hosts):
        raise ValueError(f"host_index {host_index} is outside the {len(hosts)} host cell ID(s)")
    eligible = eligible_rbs(nrb, f_ref_mod_khz)
    if rbs is None:
        chosen = {"lower": [("lower", eligible[0])], "upper": [("upper", eligible[-1])],
                  "both": [("lower", eligible[0]), ("upper", eligible[-1])]}[position]
    else:
        picked = sorted({int(r) for r in rbs})
        bad = [r for r in picked if r not in eligible]
        if not picked or bad:
            raise ValueError(f"RBs {bad or list(rbs)} are not eligible NB-IoT positions for N_RB={nrb}: {eligible}")
        chosen = [("lower" if i == 0 else "upper", r) for i, r in enumerate(picked)]
    n_id = nb_iot_n_id(hosts[host_index], host_index + 1)
    values, mask = _build_frame(n_id, True, hosts[host_index] % 6, 2, 4, 3)
    carriers = tuple(NbIotCarrierPlan(side, rb, n_id, values, mask) for side, rb in chosen)
    n_nb_re = sum(c.n_re for c in carriers)
    nr_gain, nb_gain = power_split(nrb, n_nb_re, boost_db)
    return NbIotOverlayPlan(nrb, float(boost_db), hosts[host_index], Fraction(f_ref_mod_khz).limit_denominator(1000), carriers,
                            n_nb_re, nr_gain, nb_gain)
