"""CP-OFDM reference waveform with the LTE 20 MHz numerology, and its data-aided demodulator (``ofdm-lte20-v1``).

Scope, deliberately narrow (plan S15): one fixed numerology — 15 kHz subcarrier spacing, 2048-point FFT at
30.72 MS/s, 1200 occupied subcarriers (100 resource blocks) around an empty DC subcarrier, normal cyclic
prefix (160/144 samples, 7 symbols per 0.5 ms slot), 64QAM symbols drawn uniformly from the 64 points on every occupied subcarrier of every
symbol, drawn from a seeded generator. It is a *test waveform with known symbols*: there is no physical-channel
structure (no synchronisation signals, reference signals, control or shared channels) and nothing here is a
3GPP conformance measurement. The numerology is the one of 3GPP TS 36.211 §6.12 (OFDM baseband signal
generation); the exact standard version a profile cites is fixed in ``docs/protocols/waveform-profiles.md``.

The demodulator is data-aided: it synchronises to the known waveform by cross-correlation, estimates a carrier
frequency offset from the cyclic prefix, equalises every subcarrier with a least-squares complex gain estimated
from the known symbols over the whole measurement window, and reports the RMS error vector magnitude. Every
choice that differs from a standard's EVM procedure is stated in the protocol document, not hidden.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from opendpd.core import measurement

from opendpd.schemas.waveform import (
    WaveformBinding,
    OFDM_LTE20_FFT_SIZE,
    OFDM_LTE20_OCCUPIED_SUBCARRIERS,
    OFDM_LTE20_SAMPLE_RATE_HZ,
    OFDM_LTE20_SAMPLES_PER_SUBFRAME,
    OFDM_LTE20_SUBCARRIER_SPACING_HZ,
    WaveformSpec,
)

FS = OFDM_LTE20_SAMPLE_RATE_HZ
SCS = OFDM_LTE20_SUBCARRIER_SPACING_HZ
NFFT = OFDM_LTE20_FFT_SIZE
N_SC = OFDM_LTE20_OCCUPIED_SUBCARRIERS
CP_SLOT = (160, 144, 144, 144, 144, 144, 144)        # normal cyclic prefix, samples per symbol of a slot
SYMBOLS_PER_SUBFRAME = 2 * len(CP_SLOT)
SAMPLES_PER_SUBFRAME = OFDM_LTE20_SAMPLES_PER_SUBFRAME
SCALE = NFFT / math.sqrt(N_SC)                        # ifft output -> unit average time-domain power
MIN_CORRELATION = 0.3                                 # below this the signal is not the bound waveform


def qam64_points() -> np.ndarray:
    """The 64 constellation points, unit average power (levels ±1, ±3, ±5, ±7 divided by sqrt(42))."""
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
    return (levels[:, None] + 1j * levels[None, :]).reshape(-1) / math.sqrt(42.0)


def subcarrier_bins() -> np.ndarray:
    """FFT bin of every occupied subcarrier, in subcarrier order k = -600 … -1, +1 … +600 (DC left empty)."""
    half = N_SC // 2
    negative = np.arange(-half, 0) + NFFT
    positive = np.arange(1, half + 1)
    return np.concatenate([negative, positive])


def symbol_layout(n_subframes: int) -> Tuple[np.ndarray, np.ndarray]:
    """(cp_lengths, fft_starts) of every OFDM symbol of the waveform, in samples of the reference clock."""
    cps, starts, pos = [], [], 0
    for _ in range(2 * n_subframes):            # slots
        for cp in CP_SLOT:
            cps.append(cp)
            starts.append(pos + cp)
            pos += cp + NFFT
    assert pos == n_subframes * SAMPLES_PER_SUBFRAME
    return np.asarray(cps), np.asarray(starts)


@dataclass(frozen=True)
class Waveform:
    spec: WaveformSpec
    x: np.ndarray                 # complex128 (n_samples,), unit average power
    symbols: np.ndarray           # complex128 (n_symbols, N_SC): the reference symbol on every occupied subcarrier
    cp_lengths: np.ndarray        # int (n_symbols,)
    fft_starts: np.ndarray        # int (n_symbols,): first sample of each symbol's useful part

    @property
    def period(self) -> int:
        return int(self.x.size)

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.spec.model_dump_json().encode())
        h.update(np.ascontiguousarray(self.symbols).tobytes())
        return h.hexdigest()


def generate(spec: WaveformSpec) -> Waveform:
    """Regenerate the waveform of ``spec`` (deterministic in the seed and the length)."""
    rng = np.random.default_rng(spec.seed)
    points = qam64_points()
    cps, starts = symbol_layout(spec.n_subframes)
    symbols = points[rng.integers(0, points.size, size=(spec.n_symbols, N_SC))]
    bins = subcarrier_bins()
    x = np.zeros(spec.n_samples, dtype=np.complex128)
    for l in range(spec.n_symbols):
        grid = np.zeros(NFFT, dtype=np.complex128)
        grid[bins] = symbols[l]
        useful = np.fft.ifft(grid) * SCALE
        s, cp = int(starts[l]), int(cps[l])
        x[s:s + NFFT] = useful
        x[s - cp:s] = useful[NFFT - cp:]          # the cyclic prefix is the tail of the useful part
    return Waveform(spec=spec, x=x, symbols=symbols, cp_lengths=cps, fft_starts=starts)


def to_iq(x: np.ndarray) -> np.ndarray:
    """float32 (n, 2) I/Q columns, the layout every OpenDPD dataset uses."""
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def to_complex(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128).reshape(-1)
    arr = arr.astype(np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, 0] + 1j * arr[:, 1]


# --- packages -------------------------------------------------------------------------------------------

def write_package(wf: Waveform, out: Path) -> Path:
    """A waveform package: the spec with its hash, the playable I/Q and the reference symbols."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "spec": json.loads(wf.spec.model_dump_json()),
        "sha256": wf.sha256(),
        "n_samples": wf.period,
        "sample_rate_hz": wf.spec.sample_rate_hz,
        "x_format": "float32 (n, 2) I/Q columns, unit average power, play in a loop",
        "symbols_format": "complex64 (n_symbols, 1200): occupied subcarriers k = -600..-1, +1..+600 per OFDM symbol",
        "note": "test waveform with known symbols; no physical-channel structure; not a conformance signal",
    }
    (out / "waveform.json").write_text(json.dumps(doc, indent=2))
    np.save(out / "x.npy", to_iq(wf.x))
    np.save(out / "symbols.npy", wf.symbols.astype(np.complex64))
    return out / "waveform.json"


def read_package(path: Path) -> Tuple[WaveformSpec, str]:
    """The spec and recorded hash of a package (``waveform.json`` or its directory); the waveform is regenerated."""
    path = Path(path)
    if path.is_dir():
        path = path / "waveform.json"
    doc = json.loads(path.read_text())
    spec = WaveformSpec.model_validate(doc["spec"])
    return spec, str(doc.get("sha256", ""))


# --- rate conversion ------------------------------------------------------------------------------------

def rate_ratio(fs: float) -> Optional[Tuple[int, int]]:
    """(up, down) that converts ``fs`` to the waveform clock exactly, or None when no small rational does."""
    if fs <= 0 or fs < FS:
        return None
    return measurement.rate_ratio(fs, FS)


def to_baseband_rate(y: np.ndarray, fs: float) -> Optional[np.ndarray]:
    """Bring a capture at ``fs`` to the waveform clock (polyphase resampling); None when ``fs`` is unsupported."""
    return None if rate_ratio(fs) is None else measurement.resample(y, fs, FS)


# --- data-aided demodulation ------------------------------------------------------------------------------

def synchronize(y: np.ndarray, wf: Waveform) -> Tuple[int, float]:
    """Offset τ with y[n] ≈ x[(n + τ) mod period] and the normalised correlation peak (circular, FFT based)."""
    period = wf.period
    w = np.asarray(y, dtype=np.complex128)[:period]
    padded = np.zeros(period, dtype=np.complex128)
    padded[:w.size] = w
    r = np.fft.ifft(np.fft.fft(padded) * np.conj(np.fft.fft(wf.x)))
    m = int(np.argmax(np.abs(r)))
    norm = float(np.linalg.norm(w) * np.linalg.norm(wf.x))
    corr = float(np.abs(r[m]) / norm) if norm > 0 else 0.0
    return (-m) % period, corr


@dataclass
class Demodulation:
    offset: int
    correlation: float
    cfo_hz: float
    n_symbols: int
    evm_rms_pct: float
    evm_db: float
    equalizer: np.ndarray                     # complex (N_SC,): estimated gain per subcarrier


def _symbol_windows(wf: Waveform, offset: int, n: int) -> List[Tuple[int, int]]:
    """(reference symbol index, start of its useful part in capture time) for every symbol whose cyclic prefix
    and useful part lie inside the capture of length ``n``; the waveform repeats every ``period`` samples."""
    period = wf.period
    out: List[Tuple[int, int]] = []
    for k in range((offset - period) // period - 1, (n + offset) // period + 2):
        base = k * period - offset
        for l in range(wf.spec.n_symbols):
            start = int(wf.fft_starts[l]) + base
            cp = int(wf.cp_lengths[l])
            if start - cp >= 0 and start + NFFT <= n:
                out.append((l, start))
    return out


CP_GUARD = 32   # first cyclic-prefix samples skipped by the coarse estimator: they carry the previous symbol's tail
                # through any channel memory shorter than 32 samples (about 1 µs)


def estimate_cfo(y: np.ndarray, wf: Waveform, windows: List[Tuple[int, int]]) -> float:
    """Coarse carrier frequency offset in Hz from the cyclic-prefix correlation of every usable symbol."""
    acc = 0.0 + 0.0j
    for l, start in windows:
        cp = int(wf.cp_lengths[l])
        head = y[start - cp + CP_GUARD:start]
        tail = y[start + NFFT - cp + CP_GUARD:start + NFFT]
        acc += np.vdot(tail, head)            # sum head * conj(tail)
    if acc == 0:
        return 0.0
    return float(-np.angle(acc) / (2 * math.pi) * SCS)


def _grid(y: np.ndarray, wf: Waveform, windows: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    bins = subcarrier_bins()
    received = np.empty((len(windows), N_SC), dtype=np.complex128)
    reference = np.empty_like(received)
    for i, (l, start) in enumerate(windows):
        received[i] = (np.fft.fft(y[start:start + NFFT]) / SCALE)[bins]
        reference[i] = wf.symbols[l]
    return received, reference


def _equalize(received: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares complex gain per subcarrier over the whole window (data aided), and the equalised symbols."""
    num = np.sum(received * np.conj(reference), axis=0)
    den = np.sum(np.abs(reference) ** 2, axis=0)
    equalizer = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return equalizer, received / np.where(equalizer != 0, equalizer, 1.0)


def demodulate(y: np.ndarray, wf: Waveform, *, correct_cfo: bool = True, max_iterations: int = 3) -> Optional[Demodulation]:
    """Data-aided demodulation of ``y`` (complex, at the waveform clock). None when ``y`` does not correlate
    with the waveform or holds no complete OFDM symbol; the caller turns that into an explicit status.

    Steps: circular cross-correlation with the known waveform (integer timing), coarse frequency offset from the
    cyclic prefix, then up to ``max_iterations`` rounds of {least-squares equaliser per subcarrier over the whole
    window, fine frequency offset from the slope of the per-symbol residual phase against time}. No per-symbol
    phase tracking is applied: what is left after the frequency correction counts as error."""
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    if y.size < NFFT + max(CP_SLOT):
        return None
    offset, corr = synchronize(y, wf)
    if corr < MIN_CORRELATION:
        return None
    windows = _symbol_windows(wf, offset, y.size)
    if not windows:
        return None
    n = np.arange(y.size)
    cfo = estimate_cfo(y, wf, windows) if correct_cfo else 0.0
    times = (np.array([start for _, start in windows], dtype=np.float64) + NFFT / 2) / FS
    for _ in range(max_iterations if correct_cfo else 1):
        y_c = y * np.exp(-2j * math.pi * cfo / FS * n) if cfo != 0.0 else y
        received, reference = _grid(y_c, wf, windows)
        equalizer, equalised = _equalize(received, reference)
        if not correct_cfo or len(windows) < 2:
            break
        phase = np.unwrap(np.angle(np.sum(equalised * np.conj(reference), axis=1)))
        slope = np.polyfit(times, phase, 1)[0] if np.ptp(times) > 0 else 0.0
        delta = float(slope / (2 * math.pi))
        cfo += delta
        if abs(delta) < 1e-3:
            break
    err = equalised - reference
    p_ref = float(np.sum(np.abs(reference) ** 2))
    evm = math.sqrt(float(np.sum(np.abs(err) ** 2)) / p_ref) if p_ref > 0 else float("nan")
    evm_db = 20 * math.log10(evm) if evm > 0 else -300.0
    return Demodulation(offset=offset, correlation=corr, cfo_hz=cfo, n_symbols=len(windows),
                        evm_rms_pct=evm * 100.0, evm_db=evm_db, equalizer=equalizer)


# --- binding a capture to the waveform ----------------------------------------------------------------

def bind_input(x_iq: np.ndarray, fs: float, spec: WaveformSpec, *, package_sha256: Optional[str] = None) -> WaveformBinding:
    """Correlate a dataset's *input* column (I/Q at ``fs``) with the regenerated waveform and return the binding.

    Only one waveform period of the input is examined (instruments loop the waveform), so a Stress-size capture
    costs the same as a short one. Raises ``ValueError`` when the rate cannot be converted or the input is not
    the waveform: a binding is never recorded on a guess."""
    ratio = rate_ratio(fs)
    if ratio is None:
        raise ValueError(f"capture rate {fs / 1e6:.6g} MS/s cannot be converted to the waveform clock "
                         f"({FS / 1e6:.2f} MS/s) with a small exact ratio, or is below it")
    wf = generate(spec)
    needed = int(math.ceil(wf.period * fs / FS)) + 64
    head = to_complex(np.asarray(x_iq[:needed]))
    baseband = to_baseband_rate(head, fs)
    assert baseband is not None
    offset, corr = synchronize(baseband, wf)
    if corr < MIN_CORRELATION:
        raise ValueError(f"the input does not correlate with {spec.waveform_id} seed {spec.seed} "
                         f"(normalised peak {corr:.3f}, needs {MIN_CORRELATION})")
    return WaveformBinding(spec=spec, input_offset_samples=offset, correlation=min(corr, 1.0), input_sample_rate_hz=fs,
                           package_sha256=package_sha256 or None,
                           bound_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
