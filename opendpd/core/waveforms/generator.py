"""Deterministic complex-baseband synthesis and numerical inspection.

All OFDM profiles produce uncoded continuous payloads with generic pilots.
RF carrier frequency is metadata: it never aliases a GHz carrier into baseband.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal

from opendpd.schemas.signal_generator import GeneratorAnalysis, GeneratorConfig


def qam(rng, order, shape):
    if order == 2:
        return (2*rng.integers(0, 2, shape)-1).astype(np.complex128)
    side = math.isqrt(order)
    # Uniform symbol alphabet; payload bits and FEC are intentionally not modeled.
    real = 2*rng.integers(0, side, shape) - side + 1
    imag = 2*rng.integers(0, side, shape) - side + 1
    return (real + 1j*imag) / math.sqrt(2*(order-1)/3)


def allocation(config):
    span = sum(config.channel_subcarriers) + (len(config.channel_subcarriers)-1)*config.channel_gap_bins
    # Removing zero before slicing preserves every channel's requested tone count.
    available = np.arange(-(span//2), math.ceil(span/2) + int(config.dc_null))
    if config.dc_null:
        available = available[available != 0]
    channels, cursor = [], 0
    for count in config.channel_subcarriers:
        channels.append(available[cursor:cursor+count])
        cursor += count + config.channel_gap_bins
    active = np.concatenate(channels)
    if config.pilot_mode == "explicit":
        pilots = np.array(config.pilot_indices, dtype=int)
        if not len(pilots) or not set(pilots).issubset(set(active)):
            raise ValueError("Explicit pilot indices must be nonempty and belong to active channel subcarriers.")
    elif config.pilot_mode == "comb":
        pilots = np.concatenate([bins[::config.pilot_spacing] for bins in channels])
    else:
        pilots = np.array([], dtype=int)
    if any(len(np.setdiff1d(bins, pilots)) == 0 for bins in channels):
        raise ValueError("Each channel needs at least one data subcarrier in addition to pilots.")
    return channels, pilots


def cp_length(config, index):
    if config.cp_mode == "fixed":
        return config.cp_samples * config.oversampling
    if config.cp_mode == "nr_extended":
        return config.fft_size * config.oversampling // 4
    mu = round(math.log2(config.sample_rate_hz/(config.fft_size*config.oversampling*15000)))
    # TS 38.211 §5.3.1: the long CP occurs twice per subframe, independent of μ.
    length = config.fft_size * (144 + (16*2**mu if index % (7*2**mu) == 0 else 0)) / 2048
    return round(length * config.oversampling)


def rrc_taps(sps, rolloff, span):
    time = np.arange(-span*sps, span*sps+1) / sps
    if rolloff == 0:
        taps = np.sinc(time)
    else:
        taps = np.empty_like(time)
        zero = np.isclose(time, 0)
        singular = np.isclose(np.abs(time), 1/(4*rolloff))
        regular = ~(zero | singular)
        t = time[regular]
        taps[regular] = (np.sin(np.pi*t*(1-rolloff)) + 4*rolloff*t*np.cos(np.pi*t*(1+rolloff))) / (np.pi*t*(1-(4*rolloff*t)**2))
        taps[zero] = 1 + rolloff*(4/np.pi-1)
        taps[singular] = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff)) + (1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
    return taps / np.sqrt(np.sum(taps*taps))


def synthesize(config: GeneratorConfig):
    """Return exact length complex64 IQ plus bounded diagnostic data."""
    seeds = np.random.SeedSequence(config.seed).spawn(2)
    rng, noise = [np.random.Generator(np.random.PCG64(s)) for s in seeds]
    n = config.sample_count
    fft = config.fft_size * config.oversampling
    refs, positions, cp_lengths = [], [], []
    channels, pilots = [], np.array([], dtype=int)
    complete, trailing = 0, 0
    constellation = np.array([], dtype=complex)
    if config.waveform == "ofdm":
        channels, pilots = allocation(config)
        blocks, cursor, index = [], 0, 0
        active = np.concatenate(channels)
        while cursor < n:
            grid = np.zeros(fft, dtype=np.complex128)
            for bins, order, power in zip(channels, config.channel_modulations, config.channel_power_db):
                grid[bins % fft] = qam(rng, order, len(bins)) * 10**(power/20)
                channel_pilots = np.intersect1d(pilots, bins)
                grid[channel_pilots % fft] = qam(rng, 2, len(channel_pilots)) * 10**((power+config.pilot_boost_db)/20)
            block = np.fft.ifft(grid) * fft / np.sqrt(len(active))
            cp = cp_length(config, index)
            symbol = np.r_[block[-cp:], block] if cp else block
            blocks.append(symbol[:n-cursor])
            if cursor + len(symbol) <= n:
                complete += 1
                if len(refs) < 16:
                    refs.append(grid)
                    positions.append(cursor+cp)
            else:
                trailing = n-cursor
            if len(cp_lengths) < 28:
                cp_lengths.append(cp)
            cursor += len(symbol)
            index += 1
        x = np.concatenate(blocks)
    elif config.waveform == "qam":
        sps, span = config.samples_per_symbol, config.rrc_span_symbols
        symbols = qam(rng, config.modulation_order, math.ceil(n/sps) + 4*span)
        up = np.zeros(len(symbols)*sps, dtype=complex)
        up[::sps] = symbols
        filtered = signal.fftconvolve(up, rrc_taps(sps, config.rrc_rolloff, span), mode="same")
        x = filtered[2*span*sps:2*span*sps+n]
        constellation = symbols[2*span:2*span+min(4096, math.ceil(n/sps))]
        complete, trailing = divmod(n, sps)
    else:
        time = np.arange(n) / config.sample_rate_hz
        if config.waveform == "tone":
            x = np.exp(2j*np.pi*config.tone_frequency_hz*time)
        elif config.waveform == "multitone":
            x = np.zeros(n, dtype=complex)
            for frequency, phase in zip(np.linspace(-config.bandwidth_hz/2, config.bandwidth_hz/2, config.tone_count), rng.uniform(-np.pi, np.pi, config.tone_count)):
                x += np.exp(2j*np.pi*frequency*time + 1j*phase)
        else:
            slope = config.bandwidth_hz / (n/config.sample_rate_hz)
            x = np.exp(2j*np.pi*(-config.bandwidth_hz/2*time + .5*slope*time**2))
    scale = config.rms / np.sqrt(np.mean(np.abs(x)**2))
    x *= scale
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
    x = x.astype(np.complex64)
    recovered, references = [], []
    evm_percent = None
    if refs:
        active = np.concatenate(channels)
        data = np.setdiff1d(active, pilots)
        for grid, position in zip(refs, positions):
            received = np.fft.fft(x[position:position+fft]) / (fft/np.sqrt(len(active))*scale)
            recovered.append(received[data % fft])
            references.append(grid[data % fft])
        constellation = np.concatenate(recovered)
        truth = np.concatenate(references)
        evm_percent = float(100*np.sqrt(np.sum(np.abs(constellation-truth)**2)/np.sum(np.abs(truth)**2)))
    else:
        truth = constellation.copy()
    power = np.abs(x.astype(complex))**2
    average = float(np.mean(power))
    relative = 10*np.log10(np.maximum(power / average, 1e-30))
    ccdf_x = np.linspace(0, max(16, float(np.max(relative))), 129)
    sorted_power = np.sort(relative)
    ccdf = (n-np.searchsorted(sorted_power, ccdf_x, side="right"))/n
    nperseg = min(n, max(1024, min(16384, fft*2 if config.waveform == "ofdm" else 4096)))
    frequencies, psd = signal.welch(x.astype(complex), fs=config.sample_rate_hz, window="hann", nperseg=nperseg,
                                   noverlap=nperseg//2, return_onesided=False, detrend=False, scaling="density")
    frequencies, psd = np.fft.fftshift(frequencies), np.fft.fftshift(psd)
    cumulative = np.cumsum(psd) / np.sum(psd)
    lo, hi = np.searchsorted(cumulative, [.005, .995])
    occupied = float(frequencies[min(hi, len(frequencies)-1)] - frequencies[lo] + config.sample_rate_hz/nperseg)
    # Keep contiguous early samples in the time view; strided IQ can alias the display.
    shown = min(n, 2048)
    scatter_indices = np.linspace(0, max(0, len(constellation)-1), min(4096, len(constellation)), dtype=int)
    entries = []
    spacing = config.sample_rate_hz/fft if config.waveform == "ofdm" else None
    for index, bins in enumerate(channels):
        entries.append({"channel": index+1, "subcarriers": bins.tolist(), "pilots": np.intersect1d(bins, pilots).tolist(),
                        "modulation_order": config.channel_modulations[index], "power_db": config.channel_power_db[index]})
    notes = ["SYNTHETIC complex baseband; RF carrier frequency is metadata, not digital upconversion.",
             "PAPR and CCDF use every exported sample at the selected sample rate; unsampled analog peaks are not estimated.",
             "Power is relative to unit complex RMS (0 dBFS), not calibrated watts or dBm. PSD uses Hann-window Welch, 50% overlap.",
             "99% occupied bandwidth uses the central 99% of integrated Welch power, including impairments."]
    if config.waveform == "ofdm":
        notes += ["NR / WLAN presets specify OFDM numerology only: uncoded random payload, generic BPSK pilots, no synchronization, control channels, FEC or packet preambles. Not a conformance waveform or standard test model.",
                  "Constellation and reference EVM use data carriers from the first 16 complete OFDM symbols at most; no equalization or phase/gain fit. This is a generator diagnostic, not standard EVM."]
        if not refs:
            notes.append("No complete OFDM symbol fits this capture; increase length for constellation and reference EVM.")
    if config.waveform == "qam":
        notes.append("Constellation shows transmitted modulation symbols before RRC filtering and impairments; no receiver EVM is reported.")
    if trailing:
        notes.append(f"Exact requested length retained: {trailing} samples from the final incomplete symbol. No extra samples are exported.")
    if config.preset_id.startswith("wifi8"):
        notes.append("Wi-Fi 8 / IEEE 802.11bn is an experimental numerology profile; no draft-specific UHR features are implemented.")
    analysis = GeneratorAnalysis(sample_count=n, duration_ms=n/config.sample_rate_hz*1000,
        sample_rate_hz=config.sample_rate_hz, subcarrier_spacing_hz=spacing,
        useful_symbol_us=(1e6/spacing if spacing else None), cp_lengths_samples=cp_lengths,
        complete_symbols=complete, trailing_samples=trailing, active_carriers=sum(map(len, channels)),
        data_carriers=sum(map(len, channels))-len(pilots), pilot_carriers=len(pilots),
        rms=math.sqrt(average), peak=math.sqrt(float(np.max(power))), papr_db=float(np.max(relative)),
        mean_power_dbfs=10*math.log10(average), occupied_bandwidth_99_hz=occupied,
        dc_magnitude=float(abs(np.mean(x))), evm_percent=evm_percent, evm_symbols=len(refs),
        time_us=(np.arange(shown)/config.sample_rate_hz*1e6).tolist(), time_i=x[:shown].real.tolist(),
        time_q=x[:shown].imag.tolist(), time_envelope=np.abs(x[:shown]).tolist(),
        frequency_mhz=(frequencies/1e6).tolist(), psd_dbfs_hz=(10*np.log10(np.maximum(psd, 1e-30))).tolist(),
        constellation_i=constellation[scatter_indices].real.tolist(), constellation_q=constellation[scatter_indices].imag.tolist(),
        reference_i=truth[scatter_indices].real.tolist(), reference_q=truth[scatter_indices].imag.tolist(),
        ccdf_db=ccdf_x.tolist(), ccdf_probability=ccdf.tolist(), allocation=entries, notes=notes)
    return x, analysis
