"""Calibrated-in-sample-units measurements with bounded visualization payloads."""
from __future__ import annotations

import numpy as np
from scipy import signal


def band_power(frequency, density, sample_rate, center, bandwidth):
    """Integrate fractional FFT bins, including the periodic Nyquist bin."""
    low, high = center - bandwidth / 2, center + bandwidth / 2
    if low < -sample_rate / 2 or high > sample_rate / 2:
        return None
    df = sample_rate / len(frequency)
    weight = np.maximum(0., np.minimum(frequency + df / 2, high) - np.maximum(frequency - df / 2, low))
    weight[0] += max(0., min(sample_rate / 2 + df / 2, high) - max(sample_rate / 2 - df / 2, low))
    return float(np.sum(density * weight))


def _reduce(matrix, maximum, axis):
    if matrix.shape[axis] <= maximum:
        return matrix
    return np.stack([np.mean(part, axis=axis) for part in np.array_split(matrix, maximum, axis=axis)], axis=axis)


def inspect_signal(samples, config, reference=None):
    """All scalar statistics use the selected contiguous samples, never plot decimation."""
    x = np.asarray(samples, dtype=np.complex128).copy()
    n, fs = len(x), config.sample_rate_hz
    if not 256 <= n <= 1_000_000 or not np.isfinite(x).all():
        raise ValueError("Select 256–1,000,000 finite signal samples.")
    if reference is not None and (len(reference) != n or not np.isfinite(reference).all()):
        raise ValueError("The reference must contain the same selected sample range with finite values.")
    notes = [
        "All measurements use the entire selected contiguous range. No resampling, timing recovery or automatic standard detection is applied.",
        "Power uses unit sample amplitude as its reference: 0 dBFS means mean |x|² = 1. No impedance, watts or dBm calibration is inferred.",
        "PSD is two-sided for both real and complex inputs. Real signals retain their positive and negative frequency images.",
        "I/Q scatter contains raw samples, not demodulated symbols. Eye traces use the manually selected samples/symbol and offset.",
        "No packet decoding, spectral-mask verdict or standards EVM is inferred from an arbitrary capture.",
    ]
    if config.remove_dc:
        x -= np.mean(x)
        notes.append("DC removal is enabled for the analyzed signal. Stored samples and the reference remain unchanged.")
    if config.frequency_shift_hz:
        x *= np.exp(2j * np.pi * config.frequency_shift_hz * np.arange(n) / fs)
        notes.append("A digital frequency shift is applied for analysis; energy crossing Nyquist wraps. Stored samples are unchanged.")
    power = np.abs(x) ** 2
    average = float(np.mean(power))
    peak = float(np.max(power))
    measurements = []

    def metric(key, label, value, unit="", reason=None):
        measurements.append(dict(key=key, label=label, value=value, unit=unit,
                                 reason=reason if value is None else None))

    def db(value):
        return float(10 * np.log10(value)) if value is not None and value > 0 else None

    metric("rms", "RMS amplitude", float(np.sqrt(average)))
    metric("peak", "Peak amplitude", float(np.sqrt(peak)))
    metric("power", "Mean power", db(average), "dBFS", "Zero signal power.")
    metric("papr", "PAPR", db(max(1., peak / average)) if average else None, "dB", "Zero signal power.")
    metric("dc", "DC magnitude", float(abs(np.mean(x))))
    metric("i_rms", "I RMS", float(np.sqrt(np.mean(x.real ** 2))))
    metric("q_rms", "Q RMS", float(np.sqrt(np.mean(x.imag ** 2))))
    segment = min(config.fft_size, n)
    window = signal.get_window(config.window, segment, fftbins=True)
    frequency, density = signal.welch(x, fs, window=window, nperseg=segment,
        noverlap=int(segment * config.overlap), return_onesided=False, detrend=False, scaling="density")
    frequency, density = np.fft.fftshift(frequency), np.fft.fftshift(density)
    df = fs / segment
    metric("rbw", "Equivalent noise bandwidth", float(fs * np.sum(window ** 2) / np.sum(window) ** 2), "Hz")
    metric("bin_width", "FFT bin spacing", df, "Hz")
    total = float(np.sum(density))
    occupied = None
    if total > 0:
        cumulative = np.r_[0., np.cumsum(density) / total]
        edges = np.r_[frequency - df / 2, frequency[-1] + df / 2]
        tail = (1 - config.occupied_percent / 100) / 2
        low, high = np.interp([tail, 1 - tail], cumulative, edges)
        occupied = float(high - low)
    metric("obw", f"{config.occupied_percent:g}% occupied bandwidth", occupied, "Hz", "Zero signal power.")
    main = band_power(frequency, density, fs, config.center_hz, config.bandwidth_hz)
    metric("channel_power", "Integrated channel power", db(main), "dBFS", "No power in the measurement band.")
    offset = config.adjacent_offset_hz or config.bandwidth_hz
    for label, sign in (("lower", -1), ("upper", 1)):
        adjacent = band_power(frequency, density, fs, config.center_hz + sign * offset, config.bandwidth_hz)
        value = db(adjacent / main) if main and adjacent is not None else None
        metric("acpr_" + label, label.title() + " ACPR", value, "dBc",
               "The complete adjacent band must fit inside Nyquist and both bands must have nonzero power.")
    notes.append("ACPR integrates equal-width main and adjacent bands with fractional edge bins. An out-of-range band is unavailable, never silently truncated.")
    relative = 10 * np.log10(np.maximum(power / average, 1e-30)) if average else np.zeros(n)
    ccdf_x = np.linspace(0, max(16., float(np.max(relative))), 129)
    ccdf = (n - np.searchsorted(np.sort(relative), ccdf_x, side="right")) / n if average else np.zeros(129)
    counts, edges = np.histogram(np.sqrt(power), bins=80, range=(0., float(np.sqrt(peak)) if peak > 0 else 1.))
    shown = min(2048, n)
    excerpt = x[:shown]
    inst = np.angle(excerpt[1:] * excerpt[:-1].conj()) * fs / (2 * np.pi)
    valid = (np.abs(excerpt[:-1]) > np.sqrt(peak) * 1e-6) & (np.abs(excerpt[1:]) > np.sqrt(peak) * 1e-6)
    instantaneous = [None] + [float(v) if ok else None for v, ok in zip(inst, valid)]
    notes.append("Instantaneous frequency uses adjacent-sample phase differences; zero-envelope transitions are omitted. It is not a carrier-frequency estimator for arbitrary modulation.")
    sg_segment = min(segment, 1024)
    sf, st, sp = signal.spectrogram(x, fs, window=config.window, nperseg=sg_segment,
        noverlap=sg_segment // 2, return_onesided=False, detrend=False, scaling="density", mode="psd")
    sf, sp = np.fft.fftshift(sf), np.fft.fftshift(sp, axes=0)
    # Average in linear power across every source cell before converting to dB.
    sf = _reduce(sf, 256, 0)
    st = _reduce(st, 128, 0)
    sp = _reduce(_reduce(sp, 256, 0), 128, 1)
    notes.append(f"Spectrogram uses {sg_segment}-sample {config.window} windows, 50% overlap, and power-averages into at most 256 frequency × 128 time cells.")
    sps = config.samples_per_symbol
    starts = np.arange(config.symbol_offset, max(config.symbol_offset, min(n - 2 * sps, 65536)), sps)
    starts = starts[:64]
    eyes = x[starts[:, None] + np.arange(2 * sps + 1)] if len(starts) else np.empty((0, 2 * sps + 1), complex)
    scatter = x[np.linspace(0, n - 1, min(n, 4096), dtype=int)]
    errors = []
    if reference is not None:
        ref = np.asarray(reference, dtype=complex)
        ref_power = float(np.vdot(ref, ref).real)
        gain = np.vdot(ref, x) / ref_power if ref_power and config.reference_gain_fit else 1.
        expected = gain * ref
        denominator = float(np.vdot(expected, expected).real)
        residual = np.abs(x - expected) ** 2
        relative_error = float(np.sum(residual) / denominator) if denominator else None
        metric("reference_evm", "Reference waveform RMS error", 100 * float(np.sqrt(relative_error)) if relative_error is not None else None,
               "%", "Reference power is zero.")
        metric("reference_nmse", "Reference waveform NMSE", db(max(relative_error, 1e-30)) if relative_error is not None else None,
               "dB", "Reference power is zero.")
        if denominator:
            errors = (100 * np.sqrt(residual[:shown] / (denominator / n))).tolist()
        notes.append("Reference waveform errors use exactly aligned samples without timing search; NMSE is floored at −300 dB. These are sample-domain errors, not a protocol EVM measurement.")
        if config.reference_gain_fit:
            notes.append(f"Reference scalar least-squares gain fit is enabled: {gain.real:.9g} + j({gain.imag:.9g}). This removes a constant gain and phase difference from the error.")
    return dict(sample_count=n, real_signal=bool(np.all(x.imag == 0)), measurements=measurements,
        frequency_hz=frequency.tolist(), psd_dbfs_hz=(10 * np.log10(np.maximum(density, 1e-300))).tolist(),
        time_s=((config.start_sample + np.arange(shown)) / fs).tolist(), time_i=excerpt.real.tolist(), time_q=excerpt.imag.tolist(),
        envelope=np.abs(excerpt).tolist(), instantaneous_frequency_hz=instantaneous,
        scatter_i=scatter.real.tolist(), scatter_q=scatter.imag.tolist(),
        ccdf_db=ccdf_x.tolist(), ccdf_probability=ccdf.tolist(),
        histogram_amplitude=((edges[:-1] + edges[1:]) / 2).tolist(), histogram_probability=(counts / n).tolist(),
        spectrogram_time_s=(st + config.start_sample / fs).tolist(), spectrogram_frequency_hz=sf.tolist(),
        spectrogram_dbfs_hz=(10 * np.log10(np.maximum(sp.T, 1e-300))).tolist(),
        eye_i=eyes.real.tolist(), eye_q=eyes.imag.tolist(), reference_error_percent=errors, notes=notes)
