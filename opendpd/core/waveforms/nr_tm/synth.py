"""Synthesis of a test-model signal: carriers -> common sample rate -> generic impairment tail -> analysis.

The tail (RMS scaling, burst gating, phase offset/noise, IQ imbalance, DC, frequency offset, clipping, AWGN, optional
PA-input filter) is a copy of the tail of ``opendpd.core.waveforms.generator.synthesize`` so the behaviour of the editable
GeneratorConfig fields is identical. It is copied rather than imported because that file is hashed into every signal id
and must stay byte-identical.

NR-N-TM: ``carrier_overlay`` turns the ``nb_iot.NbIotOverlayPlan`` of a host carrier into the ``grid.Overlay`` that puts the NB-IoT
carriers into the NR resource grid slot by slot and removes their resource elements from the EVM reference.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from opendpd.schemas.signal_generator import GeneratorAnalysis, GeneratorConfig

from .grid import CarrierGrid, Overlay, build_carrier_grid
from .modulate import cp_lengths, ofdm_demodulate, ofdm_modulate, phasor
from .plan import CarrierPlan, Plan, plan_of, summarize

MAX_EVM_POINTS = 4096          # per-subcarrier EVM and allocation lists are thinned to resource-block steps above this


def carrier_overlay(plan: Plan, carrier: CarrierPlan) -> Overlay | None:
    """In-band overlay of one carrier: the NB-IoT carriers of NR-N-TM (None for a carrier that hosts none).

    Per slot the whole NR content is scaled by the NR gain, the NB-IoT resource elements of the punctured RBs are replaced by the
    boosted N-TM frame (NR slot s of a 15 kHz FDD carrier is NB-IoT subframe s mod 10) and exactly those resource elements are
    reported as excluded from the PDSCH EVM reference.
    """
    host = dict(plan.nb_iot_overlays).get(carrier.index)
    if host is None:
        return None

    def overlay(slot_grid: np.ndarray, slot_in_frame: int):
        out = host.overlay_slot(slot_grid, slot_in_frame)
        excluded = np.zeros(out.shape, dtype=bool)
        for nb_carrier in host.carriers:
            excluded[12 * nb_carrier.rb:12 * (nb_carrier.rb + 1)] = nb_carrier.mask[slot_in_frame % 10].T
        return out, excluded
    return overlay


def baseband(plan: Plan) -> tuple[np.ndarray, list[tuple[CarrierPlan, CarrierGrid]]]:
    """Sum of all carriers at the common sample rate, each independently synthesised and frequency shifted.

    Carrier c is sqrt(N_c) ifft(grid) (unitary scaling, so equal resource-element amplitude means equal power spectral
    density for every subcarrier spacing), shifted by its offset exp(j 2 pi f_c n / fs). Carriers are accumulated one at a time.
    """
    recipe, fs = plan.recipe, plan.sample_rate_hz
    total = np.zeros(plan.n_samples, dtype=np.complex128)
    grids = []
    for carrier in plan.carriers:
        grid = build_carrier_grid(plan.model, carrier.nrb, carrier.scs_khz, recipe.duplex, carrier.n_cell_id, plan.subframes,
                                  recipe.data_convention, recipe.pn_scope, recipe.layers, recipe.output_port,
                                  carrier_overlay(plan, carrier))
        samples = ofdm_modulate(grid.values, carrier.mu, carrier.n_fft, gain=math.sqrt(carrier.n_fft))
        if len(samples) != plan.n_samples:
            raise RuntimeError("Internal error: carrier length differs from the capture length.")
        if carrier.offset_hz:
            samples *= phasor(carrier.offset_hz, fs, plan.n_samples)
        total += samples
        grids.append((carrier, grid))
    return total, grids


def synthesize_nr_tm(config: GeneratorConfig):
    """Return exact length complex64 IQ plus bounded diagnostic data (same contract as ``generator.synthesize``)."""
    if config.nr_test_model is None:
        raise ValueError("The configuration has no nr_test_model recipe.")
    plan = plan_of(config.nr_test_model)
    n = plan.n_samples
    if config.sample_count != n or config.sample_rate_hz != plan.sample_rate_hz:
        raise ValueError("The configuration no longer matches its test-model recipe; edit the recipe instead.")
    noise = np.random.Generator(np.random.PCG64(np.random.SeedSequence(config.seed).spawn(2)[1]))
    x, grids = baseband(plan)

    # ---- generic tail: identical to generator.synthesize ---------------------------------------------------------
    scale = config.rms / np.sqrt(np.mean(np.abs(x)**2))
    x *= scale
    if config.burst_on_samples is not None:
        position = np.arange(n) % (config.burst_on_samples + config.burst_off_samples)
        envelope = (position < config.burst_on_samples).astype(float)
        ramp = config.burst_ramp_samples
        if ramp:
            phase = np.minimum(position, config.burst_on_samples - 1 - position) / ramp
            envelope *= .5 - .5 * np.cos(np.pi * np.clip(phase, 0, 1))
        x *= envelope
    x *= np.exp(1j * np.deg2rad(config.phase_offset_deg))
    if config.phase_noise_rms_deg:
        x *= np.exp(1j * np.deg2rad(config.phase_noise_rms_deg) * noise.standard_normal(n))
    # Gain mismatch scales I only; phase mismatch rotates Q only. There is no
    # post-impairment renormalization or receiver equalization hiding these effects.
    x = x.real * 10**(config.iq_gain_db/20) + 1j*x.imag*np.exp(1j*np.deg2rad(config.iq_phase_deg))
    x += config.dc_i + 1j*config.dc_q
    if config.frequency_offset_hz:
        x *= np.exp(2j*np.pi*config.frequency_offset_hz*np.arange(n)/config.sample_rate_hz)
    if config.clip_db is not None:
        ceiling = config.rms * 10**(config.clip_db/20)
        x *= np.minimum(1, ceiling / np.maximum(np.abs(x), 1e-30))
    if config.snr_db is not None:
        noise_rms = config.rms * 10**(-config.snr_db/20)
        x += noise_rms/np.sqrt(2) * (noise.standard_normal(n) + 1j*noise.standard_normal(n))
    if config.filter_enabled:
        # A periodic-record zero-phase low-pass retains the exact requested length.
        f = np.fft.fftfreq(n, 1 / config.sample_rate_hz) - config.frequency_offset_hz
        f = (f + config.sample_rate_hz / 2) % config.sample_rate_hz - config.sample_rate_hz / 2
        edge = np.abs(f) / (config.bandwidth_hz / 2)
        transition = np.clip((edge - .96) / .04, 0, 1)
        response = .5 * (1 + np.cos(np.pi * transition))
        response[edge >= 1] = 0
        before = np.mean(np.abs(x)**2)
        x = np.fft.ifft(np.fft.fft(x) * response)
        after = np.mean(np.abs(x)**2)
        if after <= before * 1e-20:
            raise ValueError("The configured signal lies outside the baseband filter. Adjust bandwidth or disable filtering.")
        correction = np.sqrt(before / after)
        x *= correction
        scale *= correction
    x = x.astype(np.complex64)
    return x, analyse(config, plan, x, grids, scale)


# ---------------------------------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------------------------------

def _subcarrier_bins(carrier: CarrierPlan, scs_min_khz: int) -> np.ndarray:
    """Signed bin index (units of the lowest subcarrier spacing, relative to baseband DC) of every row of a carrier."""
    rows = np.arange(carrier.subcarriers) - carrier.subcarriers // 2
    hz = float(carrier.offset_hz) + rows * carrier.scs_khz * 1000.0
    return np.floor(hz / (scs_min_khz * 1000.0) + 0.5).astype(int)


def nb_iot_entry(host, bins: np.ndarray) -> dict:
    """What the allocation entry of an NB-IoT host carrier says about its NB-IoT carriers (all JSON-serialisable)."""
    frame_re = host.nrb * 12 * 140
    return {
        "boost_db": host.boost_db, "n_id": host.carriers[0].n_id, "host_cell_id": host.host_cell_id,
        "nr_gain_db": 20 * math.log10(host.nr_gain), "nb_iot_gain_db": 20 * math.log10(host.nb_gain),
        "frame_re": frame_re, "nb_iot_re": host.n_nb_re, "nr_re": frame_re - host.n_nb_re,
        "carriers": [{"side": c.side, "rb": c.rb, "first_subcarrier": 12 * c.rb, "subcarriers": bins[12 * c.rb:12 * c.rb + 12].tolist(),
                      "re_per_frame": c.n_re, "re_per_subframe": [int(m.sum()) for m in c.mask]} for c in host.carriers],
    }


def _evm_percent(error: float, reference: float) -> float:
    return float(100 * math.sqrt(error / reference)) if reference > 0 else 0.0


def analyse(config: GeneratorConfig, plan: Plan, x: np.ndarray, grids, scale: float) -> GeneratorAnalysis:
    n, fs = plan.n_samples, plan.sample_rate_hz
    scs_min = min(c.scs_khz for c in plan.carriers)
    fft = plan.fft_size * plan.oversampling_config
    xc = x.astype(np.complex128)

    # ---- ideal demodulation per carrier: undo the frequency shift, FFT every useful part, compare with the grid ----------
    recovered, references = [], []
    carrier_evm, entries = [], []
    err_sym, ref_sym = None, None
    bins_all, err_bins, ref_bins = [], [], []
    evm_symbols = 0           # OFDM symbols of ONE carrier grid (the lowest carrier, like the EVM trend); per carrier: the allocation entries
    all_bins = [_subcarrier_bins(c, scs_min) for c in plan.carriers]
    thin = sum(c.subcarriers for c in plan.carriers) > MAX_EVM_POINTS
    for (carrier, grid), bins in zip(grids, all_bins):
        z = xc * np.conj(phasor(carrier.offset_hz, fs, n)) if carrier.offset_hz else xc
        spectrum = ofdm_demodulate(z, carrier.mu, carrier.n_fft, carrier.subcarriers, carrier.slots,
                                   gain=math.sqrt(carrier.n_fft) * scale)
        mask = grid.data_mask
        rec, ref = spectrum[mask], grid.values[mask]
        recovered.append(rec)
        references.append(ref)
        err = np.abs(rec - ref) ** 2
        power = np.abs(ref) ** 2
        carrier_evm.append(_evm_percent(float(err.sum()), float(power.sum())))
        rows, cols = np.nonzero(mask)
        used = np.unique(cols)
        if carrier.index == 0:
            e_sym = np.bincount(cols, weights=err, minlength=mask.shape[1])[used]
            p_sym = np.bincount(cols, weights=power, minlength=mask.shape[1])[used]
            err_sym, ref_sym = e_sym, p_sym
            evm_symbols = int(len(used))
        e_row = np.bincount(rows, weights=err, minlength=mask.shape[0])
        p_row = np.bincount(rows, weights=power, minlength=mask.shape[0])
        if thin:                                   # resource-block steps keep the stored lists short
            e_row, p_row = e_row.reshape(-1, 12).sum(axis=1), p_row.reshape(-1, 12).sum(axis=1)
            bins_all.append(bins.reshape(-1, 12)[:, 0])
        else:
            bins_all.append(bins)
        err_bins.append(e_row)
        ref_bins.append(p_row)
        entry = {
            "channel": carrier.index + 1,
            "subcarriers": (bins[::12] if thin else bins).tolist(), "subcarrier_step": 12 if thin else 1, "pilots": [],
            "modulation_order": 2 ** plan.model.highest_qm, "power_db": 0.0,
            "scs_khz": carrier.scs_khz, "bandwidth_mhz": carrier.bandwidth_mhz, "n_rb": carrier.nrb,
            "n_cell_id": carrier.n_cell_id, "center_offset_hz": float(carrier.offset_hz), "evm_percent": carrier_evm[-1],
            "evm_symbols": int(len(used)),
        }
        host = dict(plan.nb_iot_overlays).get(carrier.index)
        if host is not None:
            entry["nb_iot"] = nb_iot_entry(host, bins)
        entries.append(entry)
    rec_all, ref_all = np.concatenate(recovered), np.concatenate(references)
    evm_percent = _evm_percent(float(np.sum(np.abs(rec_all - ref_all) ** 2)), float(np.sum(np.abs(ref_all) ** 2)))
    evm_per_symbol = (100 * np.sqrt(err_sym / np.maximum(ref_sym, 1e-300))).tolist()
    e_b, p_b, b_b = np.concatenate(err_bins), np.concatenate(ref_bins), np.concatenate(bins_all)
    powered = p_b > 1e-24
    evm_per_carrier = (100 * np.sqrt(e_b[powered] / p_b[powered])).tolist()
    evm_bins = b_b[powered].tolist()
    constellation, truth = rec_all, ref_all

    # ---- generic numerical inspection: identical to generator.synthesize ----------------------------------------------
    power = np.abs(x.astype(complex))**2
    average = float(np.mean(power))
    relative = 10*np.log10(np.maximum(power / average, 1e-30))
    ccdf_x = np.linspace(0, max(16, float(np.max(relative))), 129)
    sorted_power = np.sort(relative)
    ccdf = (n-np.searchsorted(sorted_power, ccdf_x, side="right"))/n
    nperseg = min(n, max(1024, min(16384, fft*2)))
    frequencies, psd = signal.welch(x.astype(complex), fs=config.sample_rate_hz, window="hann", nperseg=nperseg,
                                   noverlap=nperseg//2, return_onesided=False, detrend=False, scaling="density")
    frequencies, psd = np.fft.fftshift(frequencies), np.fft.fftshift(psd)
    cumulative = np.cumsum(psd) / np.sum(psd)
    lo, hi = np.searchsorted(cumulative, [.005, .995])
    occupied = float(frequencies[min(hi, len(frequencies)-1)] - frequencies[lo] + config.sample_rate_hz/nperseg)
    shown = min(n, 2048)
    scatter_indices = np.linspace(0, max(0, len(constellation)-1), min(4096, len(constellation)), dtype=int)
    spacing = config.sample_rate_hz / fft
    first = plan.carriers[0]
    cps = cp_lengths(first.mu, first.n_fft, first.slots)
    active = sum(c.subcarriers for c in plan.carriers)
    analysis = GeneratorAnalysis(
        sample_count=n, duration_ms=n/config.sample_rate_hz*1000, sample_rate_hz=config.sample_rate_hz,
        subcarrier_spacing_hz=spacing, useful_symbol_us=1e6/spacing, cp_lengths_samples=[int(c) for c in cps[:28]],
        complete_symbols=14 * first.slots, trailing_samples=0, active_carriers=active, data_carriers=active, pilot_carriers=0,
        rms=math.sqrt(average), peak=math.sqrt(float(np.max(power))), papr_db=float(np.max(relative)),
        mean_power_dbfs=10*math.log10(average), occupied_bandwidth_99_hz=occupied, dc_magnitude=float(abs(np.mean(x))),
        evm_percent=evm_percent, evm_symbols=evm_symbols, evm_per_symbol_percent=evm_per_symbol, evm_subcarrier_indices=evm_bins,
        evm_per_subcarrier_percent=evm_per_carrier,
        time_us=(np.arange(shown)/config.sample_rate_hz*1e6).tolist(), time_i=x[:shown].real.tolist(),
        time_q=x[:shown].imag.tolist(), time_envelope=np.abs(x[:shown]).tolist(),
        frequency_mhz=(frequencies/1e6).tolist(), psd_dbfs_hz=(10*np.log10(np.maximum(psd, 1e-30))).tolist(),
        constellation_i=constellation[scatter_indices].real.tolist(), constellation_q=constellation[scatter_indices].imag.tolist(),
        reference_i=truth[scatter_indices].real.tolist(), reference_q=truth[scatter_indices].imag.tolist(),
        ccdf_db=ccdf_x.tolist(), ccdf_probability=ccdf.tolist(), allocation=entries, notes=notes(config, plan),
        nr_test_model=summarize(plan))
    return analysis


def notes(config: GeneratorConfig, plan: Plan) -> list[str]:
    recipe, model = plan.recipe, plan.model
    out = ["SYNTHETIC complex baseband; RF carrier frequency is metadata, not digital upconversion.",
           "PAPR and CCDF use every exported sample at the selected sample rate; unsampled analog peaks are not estimated.",
           "Power is relative to unit complex RMS (0 dBFS), not calibrated watts or dBm. PSD uses Hann-window Welch, 50% overlap.",
           "99% occupied bandwidth uses the central 99% of integrated Welch power, including impairments."]
    out.append(f"Baseband implementation of the {model.id} definition of 3GPP TS 38.141-1 V18.14.0 clause {model.clause}: "
               f"{'PDCCH, PDSCH and DM-RS of the NR carriers plus the NB-IoT test model' if model.nb_iot else 'PDCCH, PDSCH and DM-RS only'} "
               "(no SSB, CSI-RS or PT-RS), uncoded PN23 payload scrambled per TS 38.211, normal CP, no windowing or "
               f"filtering. {'One antenna port of two layers is exported (port ' + str(1000 + recipe.output_port) + ').' if recipe.layers == 2 else 'One antenna port (PDSCH port 1000, PDCCH port 2000) is exported.'} "
               "The signal is normalised to the configured RMS; it is not a calibrated power level and not an instrument-certified conformance waveform.")
    if model.nb_iot:
        out.append(
            "NR-N-TM: the NR carrier is NR-FR1-TM1.1; per NB-IoT carrier one NR resource block is punctured and, on the resource elements "
            "that carry NB-IoT signal, replaced by the NB-IoT test model N-TM of TS 36.141 6.1.3-6.1.6 (the only NB-IoT test model): one "
            "10 ms frame (n_f = 0, normal CP) repeated every frame with NPBCH in subframe 0, NPDCCH (format 1) in 1, NPDSCH (n_RNTI 1000, "
            "QPSK) in 2-4 and 6-8, NPSS in 5, NSSS in 9 and NRS in all other subframes, all data bits zero. NB-IoT cell identity "
            "N_ID^Ncell = 97 + 6 n + max(0, m - 1) with n the rank of the host carrier among the NB-IoT hosts and m its cell identity. "
            "Conventions where the standards are silent: the NRS is sent on port 2000 only but the port-2001 positions stay reserved "
            "(the 304/200-bit budgets of TS 36.141 6.1.4 are the budgets of a two-port NRS, four host CRS ports and three control symbols), "
            "NR resource elements of the punctured RB that carry no NB-IoT signal (control symbols, host CRS positions) keep their "
            "NR-FR1-TM1.1 content, the NSSS is sent in every frame, the position of the punctured RB assumes an NR RF reference frequency "
            "that is a multiple of 100 kHz (the standard lists no table of eligible RBs), and the power is balanced over the whole 10 ms frame.")
        out.append("EVM of NR-N-TM covers the PDSCH data resource elements of the NR content only; the resource elements that carry NB-IoT signal "
                   "are excluded from the reference (the standard has no NR EVM test with NB-IoT; NB-IoT EVM belongs to TS 36.141).")
    if recipe.data_convention == "standard":
        out.append("Data convention 'standard': one PN23 stream per user (n_RNTI) and one for the PDCCH, restarted from the all-ones state in the "
                   "first slot of every radio frame (TS 38.141-1 4.9.2.3). MATLAB 5G Toolbox instead keeps one stream per PDSCH allocation "
                   "object and never restarts it; the 'matlab' convention reproduces that. Both are readings of the same text and differ "
                   "only in the random bits (and only for TM2*, TDD special slots and captures longer than one frame).")
    else:
        out.append("Data convention 'matlab': one PN23 stream per PDSCH allocation object (user x slot class x full/special slot) and one for the "
                   "PDCCH, started once and never restarted, as MATLAB R2026a 5G Toolbox does. The TS 38.141-1 text restarts the generator in the "
                   "first slot of every frame and allows one stream per user; use data_convention 'standard' for that reading.")
    if recipe.duplex == "tdd":
        scope = ("PN bits drawn for downlink symbols only" if recipe.pn_scope == "dl_symbols" or recipe.data_convention == "matlab"
                 else "PN bits drawn as if every slot were a full downlink slot")
        out.append(f"TDD pattern of TS 38.141-1 table 4.9.2.2-1 (5 ms period repeated): flexible and uplink symbols are not transmitted "
                   f"(exactly zero samples); {scope}.")
    if len(plan.carriers) > 1:
        out.append(f"{len(plan.carriers)} carriers, each its own {model.id} with N_ID^cell 1, 2, ... in ascending frequency order unless overridden, "
                   "separated by the nominal channel spacing of TS 38.104 5.4.1.2 (100 kHz raster) plus any configured sub-block gap; equal "
                   "resource-element amplitude on every carrier, i.e. equal power spectral density.")
    out.append("EVM is an ideal-demodulation diagnostic over the PDSCH data resource elements of each carrier (frequency shift undone, FFT of the "
               "useful part of every OFDM symbol, no equalization or gain/phase fit, no EVM window); it is not the receiver EVM of TS 38.141-1 "
               "clause 6.5.3. EVM trend plots refer to the lowest carrier, and evm_symbols is the number of OFDM symbols of that one carrier "
               "grid that carry PDSCH data (not carriers times symbols); the per-carrier EVM and symbol counts are in the allocation entries.")
    if config.burst_on_samples is not None:
        out.append("Burst gating follows continuous-wave RMS normalization; average power includes idle samples and ramps. AWGN and DC offsets are added after gating and may remain during idle intervals.")
    if config.phase_noise_rms_deg:
        out.append("Phase noise is independent Gaussian phase jitter with the configured RMS per sample, not a frequency-dependent oscillator noise mask.")
    if config.filter_enabled:
        out.append("Ideal PA-input filter: zero-phase periodic-record FFT low-pass, cosine transition from 96% to 100% of the nominal half-bandwidth, centered on the frequency offset. Applied after impairments; RMS preserved. Filtering can change EVM, peaks and burst edges. Welch window leakage is not the stop-band floor.")
    out.extend(plan.warnings)
    return out
