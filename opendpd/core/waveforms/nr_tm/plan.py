"""From a recipe to a plan: validation, carrier placement, sample rate, capture length and the GeneratorConfig fields.

``validate_recipe`` / ``derive`` / ``resolve`` are the public entry points. Everything here is exact arithmetic on integers
and fractions (frequencies in Hz as ``Fraction``); no waveform is computed and no ``GeneratorConfig`` is built except in
``resolve``.

Rules (clauses of TS 38.104 / TS 38.141-1 in brackets):

* N_RB of every carrier from its (subcarrier spacing, channel bandwidth) cell of table 5.3.2-1.
* Adjacent carriers are separated (F_REF to F_REF) by the nominal channel spacing of 5.4.1.2 for 100 kHz raster bands,
  S = floor((BW1 + BW2 - 2 |GB1 - GB2|) / 0.6 MHz) x 0.3 MHz, with the minimum guard bands GB of table 5.3.3-1 taken at the
  larger of the two subcarrier spacings if both bandwidths exist for it (else the smaller one, else each at its own);
  ``gap_before_mhz`` (multiples of 0.3 MHz) adds a non-contiguous sub-block gap. Occupied subcarriers must not overlap.
* F_REF sits at subcarrier 6 N_RB of the carrier (5.4.2.2). The aggregate is centred at baseband DC by its edges (5.3A.2)
  when there are several carriers; a single carrier keeps F_REF at DC (subcarrier K/2 at 0 Hz, TS 38.211 5.3.1).
* N_ID^cell = 1 for the lowest carrier, 2 for the next, ... unless overridden [4.9.2.3]; an override is a Studio extension and is warned about.
* NR-N-TM (4.9.2.2.9): the NB-IoT carriers punctured into the host carriers come from ``nb_iot`` (see its docstring): 'lower' is the
  lowest eligible RB of the lowest carrier, 'upper' the highest eligible RB of the highest carrier, 'both' one of each (two NB-IoT
  carriers of one host when there is a single carrier). Each host gets an ``NbIotHost`` (N_ID^Ncell = 97 + 6 n + max(0, m - 1) with n
  the rank of the host among the NB-IoT hosts, the frame-energy-balance gains), computed in closed form: the NB-IoT frames themselves
  are only built when a signal is synthesised (``Plan.nb_iot_overlays``). An N_ID above 503, a boost below the minimum of
  TS 38.104 Table 6.3.4.2-1 for the host bandwidth, or a boost that leaves no power for the NR resource elements is rejected here.
  The energy of a host carrier is that of a carrier without NB-IoT
  (frame-energy balance), so its NR resource elements sit ``nr_gain`` below the other carriers (-0.17 dB at 10 MHz, X = 6 dB; more for
  narrow carriers, 'both' and large X): the equal power spectral density of NRTC1 then holds for the frame average only. The recipe must
  carry the ``nb_iot`` block explicitly; no default position or boost is filled in (so the recipe states every parameter of the signal).
* Sample rate fs = 15 kHz x 2^p, the smallest that gives every carrier the FFT size of ``nfft_for_nrb`` (<= 85 % occupancy,
  >= 128) and the aggregate at most 85 % of fs; then multiplied by the oversampling factor.

Cost: a recipe is judged once. ``derive`` / ``validate_recipe`` memoise the plan (or the refusal) per canonical recipe in a bounded
LRU shared by the recipe validator, ``GeneratorConfig`` and the routes, because the public Studio validates request bodies of
untrusted size; a plan costs about a millisecond and builds no waveform and no NB-IoT frame.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from fractions import Fraction

from opendpd.schemas.nr_test_model import (
    MAX_NB_IOT_BOOST_DB, NrTestModelConfig, NrTmCarrierSummary, NrTmNbIot, NrTmSummary, NrTmUser, shortest_number,
)

from . import nb_iot
from .modulate import nfft_for_nrb
from .spec import model_levels, prb_ranges, reference_slot_allocations, rbg_size, n_target_rbg, target_rbg_indices
from .tables import (
    GAP_STEP_KHZ, GUARD_BAND_KHZ, MAX_CELL_ID, MODEL_BY_ID, MU, NB_IOT_BANDWIDTHS_MHZ, NB_IOT_SCS_KHZ, QM_NAME, STANDARD_SUBFRAMES,
    TDD_PATTERNS, ModelDef, n_rb,
)

MAX_SAMPLES = 1_000_000
MAX_SAMPLE_RATE_HZ = 2_000_000_000
MAX_FFT_SIZE = 16384
MAX_GAP_BINS = 1024           # GeneratorConfig.channel_gap_bins upper bound
MAX_OCCUPANCY = Fraction(85, 100)


# ---------------------------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CarrierPlan:
    index: int
    scs_khz: int
    mu: int
    bandwidth_mhz: int
    nrb: int
    subcarriers: int
    n_cell_id: int
    n_fft: int                      # FFT size at the common sample rate: fs / SCS
    offset_hz: Fraction             # F_REF relative to baseband DC
    low_edge_hz: Fraction           # lower edge of the lowest occupied subcarrier
    high_edge_hz: Fraction          # upper edge of the highest occupied subcarrier
    slots: int                      # slots in the capture
    slots_per_frame: int


@dataclass(frozen=True)
class NbIotHost:
    """The NB-IoT carriers of one host carrier of NR-N-TM, planned in closed form (no frame has been built)."""
    carrier: int                    # index of the host carrier
    nrb: int
    position: str                   # 'lower', 'upper' or 'both' (as seen by this host)
    rbs: tuple[int, ...]            # punctured NR resource blocks
    n_id: int                       # N_ID^Ncell shared by the NB-IoT carriers of this host
    host_cell_ids: tuple[int, ...]  # cell identities of all NB-IoT hosts, ascending frequency
    rank: int                       # 0-based rank of this host among them
    boost_db: float
    n_nb_re: int                    # resource elements per 10 ms frame that carry NB-IoT signal
    nr_gain: float                  # amplitude of the NR resource elements of this carrier
    nb_gain: float                  # amplitude of the NB-IoT resource elements


@dataclass(frozen=True)
class Plan:
    recipe: NrTestModelConfig
    model: ModelDef
    carriers: tuple[CarrierPlan, ...]
    sample_rate_hz: int
    native_sample_rate_hz: int
    oversampling: int               # factor over the native rate (recipe level)
    fft_size: int                   # GeneratorConfig fields: sample_rate = fft_size x oversampling_config x SCS_min
    oversampling_config: int
    subframes: int
    standard_subframes: int
    truncated: bool
    n_samples: int
    aggregated_bandwidth_hz: float
    bandwidth_hz: float
    channel_gap_bins: int
    warnings: tuple[str, ...]
    nb_iot_placements: tuple[tuple[int, int], ...]     # (carrier index, resource block)
    nb_iot_hosts: tuple[NbIotHost, ...] = ()

    @property
    def duplex(self) -> str:
        return self.recipe.duplex

    @functools.cached_property
    def nb_iot_overlays(self) -> tuple[tuple[int, "nb_iot.NbIotOverlayPlan"], ...]:
        """(host carrier index, overlay of its NB-IoT carriers). Builds the NB-IoT frames, so only synthesis asks for it."""
        return tuple((h.carrier, nb_iot.nb_iot_overlay_plan(h.nrb, h.position, h.boost_db, h.host_cell_ids, host_index=h.rank))
                     for h in self.nb_iot_hosts)


@dataclass(frozen=True)
class NrTmDerived:
    fields: dict
    summary: NrTmSummary
    plan: Plan


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def _frac(value) -> Fraction:
    return Fraction(f"{value:.6f}")


def _limit_text(limit: float, boost: float | None = None) -> str:
    """A limit in dB to 0.01 dB: the nearest cent, or the cent below when the nearest would not be smaller than the ``boost`` it refused (a text
    must not say that 10.6499 or 10.65 dB has to stay below 10.65 dB) or when it is the largest value on offer ('up to', no ``boost`` given)."""
    text = f"{limit:.2f}"
    ok = float(text) <= limit if boost is None else float(text) < boost
    return text if ok else f"{math.floor(limit * 100) / 100:.2f}"


def nominal_spacing_khz(scs1: int, bw1: int, scs2: int, bw2: int) -> int:
    """Nominal spacing of adjacent carriers (kHz), TS 38.104 5.4.1.2 for 100 kHz-raster bands."""
    for scs in (max(scs1, scs2), min(scs1, scs2)):
        if bw1 in GUARD_BAND_KHZ[scs] and bw2 in GUARD_BAND_KHZ[scs]:
            gb1, gb2 = _frac(GUARD_BAND_KHZ[scs][bw1]), _frac(GUARD_BAND_KHZ[scs][bw2])
            break
    else:
        gb1, gb2 = _frac(GUARD_BAND_KHZ[scs1][bw1]), _frac(GUARD_BAND_KHZ[scs2][bw2])
    return GAP_STEP_KHZ * math.floor((1000 * (bw1 + bw2) - 2 * abs(gb1 - gb2)) / (2 * GAP_STEP_KHZ))


def _edges_hz(offset_khz: Fraction, scs_khz: int, k: int) -> tuple[Fraction, Fraction]:
    """(lower edge of the lowest, upper edge of the highest) occupied subcarrier for F_REF at `offset_khz`, in Hz."""
    low = (offset_khz - Fraction((k + 1) * scs_khz, 2)) * 1000
    high = (offset_khz + Fraction((k - 1) * scs_khz, 2)) * 1000
    return low, high


def _pow2_rate(required_hz: Fraction) -> int:
    rate = 15000
    while rate < required_hz:
        rate *= 2
    return rate


def canonical_json(recipe: NrTestModelConfig) -> str:
    """The recipe as canonical JSON: the identity of a recipe (its digest is part of the preset id, its text the key of the memo)."""
    return json.dumps(recipe.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def preset_id_for(recipe: NrTestModelConfig, canonical: str | None = None) -> str:
    """nr-tm<suffix>-<duplex>-<scs>k-<bw>m-c<carriers>-<6 hex of the recipe digest>, e.g. nr-tm3p1a-fdd-30k-100m-c1-4f2a9c."""
    model = MODEL_BY_ID[recipe.model]
    scs = sorted({c.scs_khz for c in recipe.carriers})
    bandwidth = sorted({c.bandwidth_mhz for c in recipe.carriers})
    digest = hashlib.sha256((canonical or canonical_json(recipe)).encode()).hexdigest()[:6]
    scs_token = f"{scs[0]}k" if len(scs) == 1 else "mixk"
    bandwidth_token = f"{bandwidth[0]}m" if len(bandwidth) == 1 else "mixm"
    return f"nr-{model.suffix}-{recipe.duplex}-{scs_token}-{bandwidth_token}-c{len(recipe.carriers)}-{digest}"


# ---------------------------------------------------------------------------------------------------------------
# Validation + plan
# ---------------------------------------------------------------------------------------------------------------

def make_plan(recipe: NrTestModelConfig) -> Plan:
    model = MODEL_BY_ID[recipe.model]
    carriers_in = recipe.carriers
    warnings: list[str] = []

    if model.fdd_only and recipe.duplex != "fdd":
        raise ValueError(f"{model.id} is generated for FDD only. The standard also defines a TDD form (TS 38.141-1 Table 4.9.2.2-1a); "
                         "Studio does not implement NB-IoT in-band operation with TDD in this release.")
    if model.nb_iot:
        if recipe.nb_iot is None:
            raise ValueError("NR-N-TM needs the NB-IoT block (nb_iot: position 'lower', 'upper' or 'both' and boost_db from 3 to 12 dB, "
                             "at least 6 dB for host carriers up to 20 MHz). It is not filled in implicitly so that the recipe states "
                             "every parameter of the signal.")
        for i, carrier in enumerate(carriers_in):
            if carrier.scs_khz != NB_IOT_SCS_KHZ:
                raise ValueError(f"Carrier {i + 1}: NR-N-TM is generated for 15 kHz host carriers only: an NB-IoT carrier (180 kHz) is one NR "
                                 f"resource block only at 15 kHz, and TS 38.141-1 4.9.2.2.9 describes no in-band construction for a "
                                 f"{carrier.scs_khz} kHz host.")
    elif recipe.nb_iot is not None:
        raise ValueError("The NB-IoT block belongs to the NR-N-TM test model only.")
    if recipe.layers == 2 and not model.supports_layers:
        raise ValueError("Two layers (rank 2, ports 1000 and 1001) are defined for NR-FR1-TM1.1 only (time alignment error test).")
    if recipe.output_port == 1 and recipe.layers != 2:
        raise ValueError("Output port 1 exists only with two layers.")

    nrbs = []
    for i, carrier in enumerate(carriers_in):
        try:
            nrbs.append(n_rb(carrier.scs_khz, carrier.bandwidth_mhz))
        except ValueError as exc:
            raise ValueError(f"Carrier {i + 1}: {exc}") from None
        if model.nb_iot and carrier.bandwidth_mhz not in NB_IOT_BANDWIDTHS_MHZ:
            raise ValueError(f"Carrier {i + 1}: NB-IoT in-band needs a 15 kHz channel of 3-50 MHz.")
    if carriers_in[0].gap_before_mhz:
        raise ValueError("The first carrier has no gap before it; set the gap on the following carriers.")

    # carrier placement ---------------------------------------------------------------------------------------
    offsets_khz = [Fraction(0)]
    for i in range(1, len(carriers_in)):
        a, b = carriers_in[i - 1], carriers_in[i]
        extra = _frac(b.gap_before_mhz) * 1000        # the recipe holds the exact multiple (NrTmCarrier snaps it); this guards model_construct
        if extra % GAP_STEP_KHZ:
            raise ValueError(f"Carrier {i + 1}: the gap before it must be a multiple of 0.3 MHz (the least common multiple of "
                             "the 100 kHz channel raster and the subcarrier spacings).")
        spacing = nominal_spacing_khz(a.scs_khz, a.bandwidth_mhz, b.scs_khz, b.bandwidth_mhz) + extra
        offsets_khz.append(offsets_khz[-1] + spacing)
        low_b = _edges_hz(offsets_khz[i], b.scs_khz, 12 * nrbs[i])[0]
        high_a = _edges_hz(offsets_khz[i - 1], a.scs_khz, 12 * nrbs[i - 1])[1]
        if low_b < high_a:
            raise ValueError(f"Carriers {i} and {i + 1} overlap by {float(high_a - low_b) / 1000:g} kHz of occupied subcarriers; "
                             "the nominal channel spacing of TS 38.104 5.4.1.2 does not leave room for this combination.")
    edges = [_edges_hz(offsets_khz[i], c.scs_khz, 12 * nrbs[i]) for i, c in enumerate(carriers_in)]
    shift_hz = Fraction(0)
    if len(carriers_in) > 1:
        shift_hz = -(edges[0][0] + edges[-1][1]) / 2          # centre the aggregate by its edges
    offsets_hz = [o * 1000 + shift_hz for o in offsets_khz]
    edges = [(lo + shift_hz, hi + shift_hz) for lo, hi in edges]
    span_hz = edges[-1][1] - edges[0][0]
    extent_hz = 2 * max(abs(edges[0][0]), abs(edges[-1][1]))

    # sample rate, oversampling, capture length ---------------------------------------------------------------
    required = max(Fraction(nfft_for_nrb(nrbs[i]) * c.scs_khz * 1000) for i, c in enumerate(carriers_in))
    required = max(required, span_hz / MAX_OCCUPANCY)
    native = _pow2_rate(required)
    standard_sf = STANDARD_SUBFRAMES[recipe.duplex]

    def per_sf(rate: int) -> int:                                               # samples per 1 ms subframe
        return rate // 1000

    if per_sf(native) > MAX_SAMPLES:
        raise ValueError(
            f"The aggregated signal ({float(span_hz) / 1e6:.1f} MHz) needs a sample rate of {native / 1e6:.2f} MS/s, i.e. "
            f"{per_sf(native):,} samples per subframe; at most {MAX_SAMPLES:,} samples can be generated. Use fewer or narrower carriers.")
    if recipe.oversampling is None:        # the largest factor at which the standard duration (or a longer requested capture) fits
        wanted = max(standard_sf, recipe.subframes or 0)
        oversampling = next((o for o in (4, 2, 1) if wanted * per_sf(native * o) <= MAX_SAMPLES and native * o <= MAX_SAMPLE_RATE_HZ), 1)
    else:
        oversampling = recipe.oversampling
        if native * oversampling > MAX_SAMPLE_RATE_HZ or per_sf(native * oversampling) > MAX_SAMPLES:
            raise ValueError(f"Oversampling {oversampling} would need {native * oversampling / 1e6:.2f} MS/s, i.e. "
                             f"{per_sf(native * oversampling):,} samples per subframe (limit {MAX_SAMPLES:,}). Choose a smaller factor.")
    fs = native * oversampling
    most = MAX_SAMPLES // per_sf(fs)
    if recipe.subframes is None:
        subframes = min(standard_sf, most)
    else:
        subframes = recipe.subframes
        if subframes > most:
            raise ValueError(f"{subframes} subframes at {fs / 1e6:.2f} MS/s are {subframes * per_sf(fs):,} samples; at most {MAX_SAMPLES:,} "
                             f"fit, i.e. {most} subframe{'s' if most != 1 else ''}. Reduce the capture length, the oversampling or the carriers.")
    truncated = subframes < standard_sf
    n_samples = subframes * per_sf(fs)
    if recipe.subframes is None and truncated:
        warnings.append(f"The standard duration of {standard_sf} ms ({'one radio frame' if standard_sf == 10 else 'two radio frames'}) "
                        f"would need {standard_sf * per_sf(fs):,} samples; the capture is shortened to {subframes} ms ({n_samples:,} samples, "
                        f"limit {MAX_SAMPLES:,}). Lower the oversampling or the bandwidth, or choose fewer subframes explicitly.")
    elif truncated:
        warnings.append(f"Capture length of {subframes} ms is shorter than the standard {standard_sf} ms.")
    if recipe.duplex == "tdd" and truncated and subframes < 5:
        warnings.append("Shorter than one 5 ms TDD pattern: the downlink/uplink pattern is incomplete.")

    # GeneratorConfig description -----------------------------------------------------------------------------
    scs_min = min(c.scs_khz for c in carriers_in)
    native_min = native // (scs_min * 1000)
    if native_min <= MAX_FFT_SIZE:
        fft_size, oversampling_config = native_min, oversampling
    else:
        fft_size, oversampling_config = MAX_FFT_SIZE, fs // (scs_min * 1000) // MAX_FFT_SIZE
    spacing_hz = scs_min * 1000
    ks = [12 * n for n in nrbs]
    gap_bins = 0
    if len(ks) > 1:      # a description only (the recipe is the truth; a test-model configuration skips the OFDMA span checks)
        gap_bins = min(MAX_GAP_BINS, max(0, (math.floor(span_hz / spacing_hz) - sum(ks)) // (len(ks) - 1)))
    span_bins = sum(ks) + (len(ks) - 1) * gap_bins
    outer = max(span_bins // 2, math.ceil(span_bins / 2) - 1)
    bandwidth_hz = float(min(max(extent_hz, Fraction((2 * outer + 1) * spacing_hz)), fs))

    # carriers ------------------------------------------------------------------------------------------------
    cell_ids = []
    for i, carrier in enumerate(carriers_in):
        cell_ids.append(carrier.n_cell_id if carrier.n_cell_id is not None else i + 1)
    if any(cell_ids[i] > MAX_CELL_ID for i in range(len(cell_ids))):
        raise ValueError(f"Cell identities run from 0 to {MAX_CELL_ID}; the automatic rule N_ID = 1, 2, 3, ... exceeded it.")
    for i, carrier in enumerate(carriers_in):
        if carrier.n_cell_id is not None and carrier.n_cell_id != i + 1:
            warnings.append(f"Carrier {i + 1}: cell identity {carrier.n_cell_id} differs from the rule of TS 38.141-1 4.9.2.3 "
                            f"(N_ID^cell = n for the n-th lowest carrier, here {i + 1}); this is a Studio extension.")
    for i in range(len(cell_ids)):
        for j in range(i + 1, len(cell_ids)):
            if cell_ids[i] == cell_ids[j] and (carriers_in[i].scs_khz, nrbs[i]) == (carriers_in[j].scs_khz, nrbs[j]):
                warnings.append(f"Carriers {i + 1} and {j + 1} share the cell identity {cell_ids[i]} and the same numerology and "
                                "bandwidth: they carry identical data (the standard gives each carrier its own N_ID^cell).")
    carrier_plans = tuple(
        CarrierPlan(i, c.scs_khz, MU[c.scs_khz], c.bandwidth_mhz, nrbs[i], ks[i], cell_ids[i], fs // (c.scs_khz * 1000),
                    offsets_hz[i], edges[i][0], edges[i][1], subframes * 2 ** MU[c.scs_khz], 10 * 2 ** MU[c.scs_khz])
        for i, c in enumerate(carriers_in))

    if recipe.pn_scope != "dl_symbols":
        if recipe.data_convention == "matlab":
            warnings.append("pn_scope has no effect with the MATLAB data convention (the PN streams run over downlink symbols only).")
        elif recipe.duplex == "fdd":
            warnings.append("pn_scope only matters for TDD (FDD has no uplink symbols).")
    if len({c.scs_khz for c in carriers_in}) > 1:
        warnings.append("Carriers with different subcarrier spacings are summed; each follows its own numerology and the slot "
                        "grids align at subframe boundaries. The ideal-demodulation EVM of each carrier then includes the "
                        "inter-numerology leakage of the other carriers.")
        if recipe.duplex == "tdd":
            first_scs = carriers_in[0].scs_khz
            others = "; ".join(f"{scs} kHz: {' '.join(TDD_PATTERNS[scs].labels())}" for scs in sorted({c.scs_khz for c in carriers_in} - {first_scs}))
            warnings.append(f"TDD with different subcarrier spacings: tdd_pattern lists the pattern of carrier 1 ({first_scs} kHz); the other "
                            f"carriers follow the pattern of their own numerology ({others}).")

    placements: tuple[tuple[int, int], ...] = ()
    nb_hosts: tuple[NbIotHost, ...] = ()
    if model.nb_iot:
        nb_hosts, nb_warnings = _plan_nb_iot(recipe.nb_iot, [c.bandwidth_mhz for c in carriers_in], nrbs, cell_ids)
        warnings.extend(nb_warnings)
        placements = tuple((host.carrier, rb) for host in nb_hosts for rb in host.rbs)

    return Plan(recipe, model, carrier_plans, fs, native, oversampling, fft_size, oversampling_config, subframes, standard_sf,
                truncated, n_samples, float(span_hz), bandwidth_hz, gap_bins, tuple(warnings), placements, nb_hosts)


def _plan_nb_iot(nb: NrTmNbIot, bandwidths: list[int], nrbs: list[int], cell_ids: list[int]):
    """The NB-IoT hosts of NR-N-TM (one per host carrier) plus their warnings (the power convention), in closed form.

    'lower' = L_NB-IoT of the lowest carrier, 'upper' = R_NB-IoT of the highest carrier, 'both' = one of each. A host carrier is
    an NB-IoT host for the NB-IoT cell identity rule: its rank n among the hosts, ascending in frequency, enters N_ID^Ncell.
    Nothing here builds an NB-IoT frame: the number of resource elements that carry NB-IoT signal depends on the cell identities only
    through their residue modulo 6 (``nb_iot.frame_re_count``), which is all the power budget needs.
    """
    last = len(nrbs) - 1
    wanted = ([(0, "lower")] if nb.position in ("lower", "both") else []) + ([(last, "upper")] if nb.position in ("upper", "both") else [])
    hosts = sorted({carrier for carrier, _ in wanted})
    host_ids = tuple(cell_ids[carrier] for carrier in hosts)
    planned, warnings = [], []
    for rank, host in enumerate(hosts):
        sides = [side for carrier, side in wanted if carrier == host]
        position = "both" if len(sides) == 2 else sides[0]
        try:
            n_id = nb_iot.nb_iot_n_id(cell_ids[host], rank + 1)
        except ValueError as exc:
            raise ValueError(f"Carrier {host + 1}: {exc}. Give this carrier a smaller n_cell_id.") from None
        nrb = nrbs[host]
        eligible = nb_iot.eligible_rbs(nrb)
        rbs = tuple({"lower": [eligible[0]], "upper": [eligible[-1]], "both": [eligible[0], eligible[-1]]}[position])
        floor = max(nb_iot.power_dynamic_range_floor_db(bandwidths[host], nrb, rb) for rb in rbs)
        if nb.boost_db < floor:
            raise ValueError(
                f"Carrier {host + 1}: the NB-IoT power boost X of a {bandwidths[host]} MHz host carrier must be at least {floor} dB "
                f"(TS 38.104 Table 6.3.4.2-1, declared per TS 38.141-1 D.43: +6 dB for hosts of 3 to 20 MHz at the positions L_NB-IoT and "
                f"R_NB-IoT, +3 dB for 25 MHz and wider); got {shortest_number(nb.boost_db)} dB. Studio accepts up to {MAX_NB_IOT_BOOST_DB} dB (a Studio limit).")
        n_nb = len(rbs) * nb_iot.frame_re_count(n_id, host_cell_id=cell_ids[host])
        limit = nb_iot.max_boost_db(nrb, n_nb)
        if nb.boost_db >= limit:
            hint = (f" or use a single NB-IoT carrier (position 'lower' or 'upper', up to {_limit_text(min(nb_iot.max_boost_db(nrb, n_nb // 2), MAX_NB_IOT_BOOST_DB))} dB)"
                    if position == "both" else "")
            raise ValueError(
                f"Carrier {host + 1}: an NB-IoT boost of {shortest_number(nb.boost_db)} dB is not possible with position '{position}' on a {nrb}-RB "
                f"carrier: {n_nb:,} of its {nrb * 12 * 140:,} resource elements per 10 ms frame carry NB-IoT signal, so the boost must "
                f"stay below {_limit_text(limit, nb.boost_db)} dB to leave power for the NR resource elements. Reduce boost_db{hint}.")
        nr_gain, nb_gain = nb_iot.power_split(nrb, n_nb, nb.boost_db)
        planned.append(NbIotHost(host, nrb, position, rbs, n_id, host_ids, rank, float(nb.boost_db), n_nb, nr_gain, nb_gain))
        nr_db, nb_db = 20 * math.log10(nr_gain), 20 * math.log10(nb_gain)
        warnings.append(
            f"NR-N-TM, carrier {host + 1}: NB-IoT carrier{'s' if len(rbs) == 2 else ''} at RB {', '.join(str(rb) for rb in rbs)} "
            f"(N_ID^Ncell {n_id}), {n_nb:,} of {nrb * 12 * 140:,} resource elements per 10 ms frame carry NB-IoT signal at {nb_db:+.2f} dB "
            f"(the X = {shortest_number(nb.boost_db)} dB boost over the average resource element); the NR resource elements of this carrier are "
            f"{nr_db:+.3f} dB relative to an unboosted carrier. TS 38.141-1 4.9.2.2.9 gives the power formulas but no counting window: the "
            "energy of the whole 10 ms frame is balanced, so the average power of the carrier equals that of a carrier without NB-IoT.")
    if len(nrbs) > 1:
        warnings.append("NR-N-TM with several carriers: only the host carriers carry NB-IoT; the NR resource elements of a host are below "
                        "the level of the other carriers by the figure given for that host, so the equal power spectral density of NRTC1 "
                        "holds for the frame average of each carrier, not for its NR resource elements.")
    return tuple(planned), warnings


# ---------------------------------------------------------------------------------------------------------------
# Summary and GeneratorConfig fields
# ---------------------------------------------------------------------------------------------------------------

def _carrier_summary(plan: Plan, cp: CarrierPlan) -> NrTmCarrierSummary:
    model = plan.model
    levels = model_levels(model, cp.nrb)
    users: list[NrTmUser] = []
    references = reference_slot_allocations(model.id, cp.nrb)
    for user in model.users:
        ranges: list[list[int]] = []
        count = 0
        for allocations in references:
            for alloc in allocations:
                if alloc.rnti == user.rnti:
                    ranges.extend(prb_ranges(alloc.prbs))
                    count = len(alloc.prbs)
        users.append(NrTmUser(rnti=user.rnti, role=user.role, modulation=QM_NAME[user.qm], prbs=count, rb_ranges=ranges,
                              start_symbol=user.start_symbol, level_db=float(levels[user.rnti]), evm_measured=user.evm_measured))
    rbg = model.structure == "rbg"
    return NrTmCarrierSummary(
        index=cp.index, scs_khz=cp.scs_khz, bandwidth_mhz=cp.bandwidth_mhz, nrb=cp.nrb, subcarriers=cp.subcarriers, n_fft=cp.n_fft,
        n_cell_id=cp.n_cell_id, center_offset_hz=float(cp.offset_hz), occupied_bandwidth_hz=float(cp.subcarriers * cp.scs_khz * 1000),
        slots=cp.slots, slots_per_frame=cp.slots_per_frame,
        rbg_size=rbg_size(cp.nrb - 3) if rbg else None, n_rbg=n_target_rbg(cp.nrb, model.x_percent) if rbg else None,
        boosted_rbgs=list(target_rbg_indices(cp.nrb, model.x_percent)) if rbg else [], users=users)


def summarize(plan: Plan) -> NrTmSummary:
    first = plan.carriers[0]
    tdd = TDD_PATTERNS[first.scs_khz].labels() if plan.duplex == "tdd" else None
    return NrTmSummary(
        model=plan.model.id, carriers=[_carrier_summary(plan, c) for c in plan.carriers], sample_rate_hz=float(plan.sample_rate_hz),
        native_sample_rate_hz=float(plan.native_sample_rate_hz), oversampling=plan.oversampling, subframes=plan.subframes,
        standard_subframes=plan.standard_subframes, truncated=plan.truncated, n_samples=plan.n_samples,
        duration_ms=plan.n_samples / plan.sample_rate_hz * 1000, aggregated_bandwidth_hz=plan.aggregated_bandwidth_hz,
        tdd_pattern=tdd, nb_iot_rbs=[rb for _, rb in plan.nb_iot_placements] if plan.nb_iot_placements else None,
        warnings=list(plan.warnings))


# The GeneratorConfig fields that a test-model configuration derives from its recipe. filter_enabled is only a default
# (the user may switch the PA-input filter on); every other entry must equal the derivation.
STRUCTURAL_FIELDS = ("preset_id", "waveform", "sample_rate_hz", "bandwidth_hz", "length_mode", "n_samples", "fft_size", "oversampling",
                     "channel_subcarriers", "channel_modulations", "channel_power_db", "shared_channel_settings", "channel_gap_bins",
                     "dc_null", "pilot_mode", "cp_mode", "filter_enabled")


def structural_fields(plan: Plan, canonical: str | None = None) -> dict:
    qms = [2 ** plan.model.highest_qm] * len(plan.carriers)
    counts = [c.subcarriers for c in plan.carriers]
    fields = {
        "preset_id": preset_id_for(plan.recipe, canonical),
        "waveform": "ofdm",
        "sample_rate_hz": float(plan.sample_rate_hz),
        "bandwidth_hz": plan.bandwidth_hz,
        "length_mode": "samples",
        "n_samples": plan.n_samples,
        "fft_size": plan.fft_size,
        "oversampling": plan.oversampling_config,
        "channel_subcarriers": counts,
        "channel_modulations": qms,
        "channel_power_db": [0.0] * len(counts),
        "shared_channel_settings": len(set(counts)) == 1 and len(set(qms)) == 1,
        "channel_gap_bins": plan.channel_gap_bins,
        "dc_null": False,
        "pilot_mode": "none",
        "cp_mode": "nr_normal",
        "filter_enabled": False,
    }
    return fields


# ---------------------------------------------------------------------------------------------------------------
# Memo: a recipe is judged once
# ---------------------------------------------------------------------------------------------------------------

MEMO_SIZE = 256


@dataclass
class _Derivation:
    plan: Plan                       # holds a private copy of the recipe; callers get it with their own recipe swapped in
    fields: dict                     # never handed out: callers get a deep copy
    summary: NrTmSummary | None = None   # built on first use (validation does not need it)


_memo: "OrderedDict[str, _Derivation | str]" = OrderedDict()      # canonical recipe -> derivation, or the text of its refusal
_memo_lock = threading.Lock()


def _derivation(recipe: NrTestModelConfig) -> _Derivation:
    """Plan and structural fields of a recipe from the bounded LRU memo (thread-safe), or ValueError with the memoised refusal.

    The recipe validator, ``GeneratorConfig`` and the routes all ask for the same recipe within one request, and untrusted bodies may
    carry many different ones, so the answer is computed once per canonical recipe and every later caller pays a dictionary lookup.
    """
    key = canonical_json(recipe)
    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None:
            _memo.move_to_end(key)
    if hit is None:
        try:
            plan = make_plan(recipe)
            hit = _Derivation(replace(plan, recipe=recipe.model_copy(deep=True)), structural_fields(plan, key))
        except ValueError as exc:
            hit = str(exc)
        with _memo_lock:
            _memo[key] = hit
            _memo.move_to_end(key)
            while len(_memo) > MEMO_SIZE:
                _memo.popitem(last=False)
    if isinstance(hit, str):
        raise ValueError(hit)
    return hit


def clear_memo() -> None:
    with _memo_lock:
        _memo.clear()


def validate_recipe(recipe: NrTestModelConfig) -> None:
    """Raise ValueError (user-readable) if the recipe violates a rule of the standard or a generator limit."""
    _derivation(recipe)


def plan_of(recipe: NrTestModelConfig) -> Plan:
    """The (memoised) plan of a recipe, carrying ``recipe`` itself."""
    return replace(_derivation(recipe).plan, recipe=recipe)


def derived_fields(recipe: NrTestModelConfig) -> dict:
    """The structural GeneratorConfig fields of a recipe (a fresh copy)."""
    return copy.deepcopy(_derivation(recipe).fields)


def derive(recipe: NrTestModelConfig) -> NrTmDerived:
    """Structural fields, summary and plan of a recipe; the objects returned are the caller's own (the memo keeps its copies)."""
    entry = _derivation(recipe)
    plan = replace(entry.plan, recipe=recipe)
    if entry.summary is None:
        entry.summary = summarize(plan)
    return NrTmDerived(copy.deepcopy(entry.fields), entry.summary.model_copy(deep=True), plan)


def resolve(recipe: NrTestModelConfig, **overrides):
    """The complete, validated GeneratorConfig of a recipe plus its summary (``overrides`` may set editable fields)."""
    from opendpd.schemas.nr_test_model_api import NrTmResolution
    from opendpd.schemas.signal_generator import GeneratorConfig

    derived = derive(recipe)
    config = GeneratorConfig(nr_test_model=recipe, **{**derived.fields, **overrides})
    return NrTmResolution(recipe=recipe, config=config, summary=derived.summary)
