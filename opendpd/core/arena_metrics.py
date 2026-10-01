"""Arena evm-aclr-v2: symbol EVM and emitted-output adjacent-channel leakage.

EVM is data-aided: the known input is the reference, so no symbol decision is
needed. Complete OFDM symbols inside the evaluated split are transformed on
their own FFT grid and compared on the occupied subcarriers, after one common
complex gain. Missing symbol metadata or incomplete symbols fail explicitly;
band-integrated waveform error is not a substitute for demodulated EVM.

Adjacent-error ratios are custom error-power ratios, equal to ACLR wherever the
reference itself is spectrally clean. Welch settings and half-open carrier bands
match opendpd-spectral-v2. Always form the complex error before taking its PSD:
subtracting two PSDs is wrong.
"""

from __future__ import annotations

import numpy as np

from opendpd.core.metrics import evaluate
from opendpd.core.metrics.general_v1 import band_power, db, psd, to_complex

VERSION = "evm-aclr-v3"


def _iq(value):
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Arena metrics require real IQ arrays with shape (samples, 2)")
    return array.astype(np.float64, copy=False)


def adjacent_error_ratios(prediction, reference, signal):
    y, r = to_complex(_iq(prediction)), to_complex(_iq(reference))
    if y.shape != r.shape or not np.isfinite(y).all() or not np.isfinite(r).all():
        raise ValueError("AER requires matching finite prediction/reference waveforms")
    fs, bw, count, size = (
        float(signal.sample_rate_hz),
        float(signal.bandwidth_hz),
        int(signal.n_sub_ch),
        int(signal.nperseg),
    )
    if len(r) < size or bw / 2 + bw / count > fs / 2:
        raise ValueError(
            "AER bands or waveform length are outside the declared capture"
        )
    width = bw / count
    f, p_ref = psd(r, fs, size)
    _, p_error = psd(y - r, fs, size)
    denominator = max(
        band_power(f, p_ref, -bw / 2 + i * width, -bw / 2 + (i + 1) * width, fs, size)[
            0
        ]
        for i in range(count)
    )
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("AER reference has no in-band carrier energy")
    result = {}
    for side, limits in (
        ("l", (-bw / 2 - width, -bw / 2)),
        ("r", (bw / 2, bw / 2 + width)),
    ):
        error, bins = band_power(f, p_error, *limits, fs, size)
        if bins == 0:
            raise ValueError("AER adjacent band has no FFT bins")
        result[f"aer_{side}_db"] = db(error / denominator)
    return result


def symbol_windows(grid, start, length):
    """(offset, size) of every demodulation window inside a split that begins at capture sample ``start``."""
    useful = grid.get("useful_samples")
    if not isinstance(useful, int) or useful <= 0:
        raise ValueError("EVM requires a declared complete OFDM symbol grid")
    period = useful + grid["prefix_samples"]
    first = -(-(start - grid["prefix_samples"]) // period)
    last = (start + length) // period - 1
    windows = [
        (k * period + grid["prefix_samples"] - start, useful)
        for k in range(first, last + 1)
    ]
    if not windows:
        raise ValueError("EVM needs one complete OFDM symbol inside the evaluated split")
    return windows


def evm_db(prediction, reference, signal, grid, start=0, window=None):
    """20·log10 of the RMS error vector over the occupied subcarriers of every demodulation window."""
    y, r = to_complex(_iq(prediction)), to_complex(_iq(reference))
    if y.shape != r.shape or not np.isfinite(y).all() or not np.isfinite(r).all():
        raise ValueError("EVM requires matching finite prediction/reference waveforms")
    fs = float(signal.sample_rate_hz)
    measured, wanted = [], []
    first, last = (0, len(r)) if window is None else window
    if not 0 <= first < last <= len(r):
        raise ValueError("EVM metric window is outside the capture")
    if grid.get("carriers"):
        # Independently timed APA carriers need their own receiver. Isolate on
        # the full held-out input, including its real context, before cutting
        # the useful symbol. The same fixed linear filter acts on y and r.
        size = grid["useful_samples"]
        time = (np.arange(len(r)) + int(start)) / fs
        frequency = np.fft.fftfreq(len(r), 1 / fs)
        for carrier in grid["carriers"]:
            oscillator = np.exp(2j * np.pi * carrier["frequency_shift_hz"] * time)
            mask = np.abs(frequency) <= carrier["filter_half_bandwidth_hz"]
            yy = np.fft.ifft(np.fft.fft(y * oscillator) * mask)
            rr = np.fft.ifft(np.fft.fft(r * oscillator) * mask)
            spans = [position - int(start) for position in carrier["fft_starts"]
                     if start + first <= position and position + size <= start + last]
            if not spans:
                raise ValueError("EVM needs one complete OFDM symbol for every measured carrier")
            bins = np.asarray(carrier["occupied_bins"], dtype=int)
            for offset in spans:
                measured.append(np.fft.fft(yy[offset:offset + size])[bins])
                wanted.append(np.fft.fft(rr[offset:offset + size])[bins])
    else:
        for offset, size in symbol_windows(grid, int(start) + first, last - first):
            offset += first
            f = np.fft.fftfreq(size, 1 / fs)
            occupied = np.zeros(size, dtype=bool)
            if "occupied_bins" in grid:
                occupied[np.asarray(grid["occupied_bins"], dtype=int)] = True
            else:
                for lo, hi in grid["occupied_hz"]:
                    occupied |= (f >= lo) & (f < hi)
            measured.append(np.fft.fft(y[offset : offset + size])[occupied])
            wanted.append(np.fft.fft(r[offset : offset + size])[occupied])
    measured, wanted = np.concatenate(measured), np.concatenate(wanted)
    reference_power = float(np.vdot(wanted, wanted).real)
    if len(wanted) == 0 or not np.isfinite(reference_power) or reference_power <= 0:
        raise ValueError("EVM reference has no occupied subcarrier energy")
    gain = np.vdot(wanted, measured) / reference_power
    signal_power = float(abs(gain) ** 2 * reference_power)
    if signal_power <= 1e-24 * reference_power:
        return 0.0  # no correlated output: the error vector is the whole reference
    error = measured - gain * wanted
    return db(float(np.vdot(error, error).real) / signal_power)


def quality_db(scores, baseline):
    """In band and out of band weigh the same: EVM gain and the gain of the worse adjacent side."""
    evm = baseline["evm_db"] - scores["evm_db"]
    adjacent = max(baseline["aclr_l_db"], baseline["aclr_r_db"]) - max(
        scores["aclr_l_db"], scores["aclr_r_db"]
    )
    return 0.5 * (evm + adjacent)


def validation_metrics(prediction, reference):
    """Validation on raw capture samples; no incomplete-symbol EVM proxy."""
    y, r = _iq(prediction), _iq(reference)
    if y.shape != r.shape or not np.isfinite(y).all() or not np.isfinite(r).all():
        raise ValueError("Validation requires matching finite prediction/reference waveforms")
    energy = float(np.sum(r * r))
    if energy <= 0:
        raise ValueError("Validation reference has no energy")
    return dict(nmse_db=db(float(np.sum((y - r)**2)) / energy),
                power_error_db=db(float(np.sum(y*y)) / energy))


def compute(prediction, reference, signal, grid, start=0, window=None):
    full_y, full_r = _iq(prediction), _iq(reference)
    first, last = (0, len(full_r)) if window is None else window
    y, r = full_y[first:last], full_r[first:last]
    scores = {m.name: m.value for m in evaluate("opendpd-spectral-v2", y, r, signal)}
    ref_scores = {
        m.name: m.value for m in evaluate("opendpd-spectral-v2", r, None, signal)
    }
    if any(scores.get(name) is None for name in ("NMSE", "IBE", "ACLR_L", "ACLR_R")):
        raise ValueError("Signal metadata does not support Arena scoring")
    power = float(10 * np.log10(np.sum(y * y) / np.sum(r * r)))
    result = dict(
        nmse_db=float(scores["NMSE"]),
        ib_error_db=float(scores["IBE"]),
        evm_db=float(evm_db(full_y, full_r, signal, grid, start, window)),
        aclr_l_db=float(scores["ACLR_L"]),
        aclr_r_db=float(scores["ACLR_R"]),
        reference_aclr_l_db=float(ref_scores["ACLR_L"]),
        reference_aclr_r_db=float(ref_scores["ACLR_R"]),
        power_error_db=power,
        **adjacent_error_ratios(y, r, signal),
    )
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite Arena waveform metric")
    return result
