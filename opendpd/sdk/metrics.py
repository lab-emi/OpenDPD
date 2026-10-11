"""Metrics and the reference waveform for callers that hold arrays and no run (``opendpd.metrics.*`` in MATLAB).

Everything here calls the implementation the run service uses (``opendpd.core.metrics.evaluate`` and
``opendpd.core.waveforms.ofdm``): there is no second implementation to keep in step and no server or project is
needed. I/Q keeps its source precision (nothing is rounded to float32), and nothing is guessed: a metric whose
inputs are missing comes back with a status and a reason, not a number. Importing this module imports nothing heavy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

LTE_PROFILE = "ofdm-lte20-evm-v1"
PROFILES = (LTE_PROFILE, "opendpd-spectral-v2", "general-spectral-v1")


def _iq(value, name: str):
    """A finite real ``(n, 2)`` array, or a complex vector, as a float array; precision is preserved."""
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype.kind == "c":
        if arr.ndim != 1 and not (arr.ndim == 2 and 1 in arr.shape):
            raise ValueError(f"{name}: complex input must be a vector, got {arr.shape}")
        arr = np.column_stack((arr.reshape(-1).real, arr.reshape(-1).imag))
    if arr.dtype.kind != "f" or arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name}: provide a nonempty single/double complex vector or an (N, 2) I/Q array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: samples must be finite")
    return np.ascontiguousarray(arr)


def lte_waveform(seed: int = 1, n_subframes: int = 10) -> Dict[str, Any]:
    """The ``ofdm-lte20-v1`` test waveform of ``seed`` and ``n_subframes``, regenerated bit for bit.

    ``iq`` is the playable float32 ``(n, 2)`` waveform as an instrument plays it (unit average power);
    ``symbols_iq`` is the reference symbol on every occupied subcarrier, ``(n_symbols * 1200, 2)`` float32, row
    ``l * 1200 + k`` being subcarrier ``k`` of OFDM symbol ``l``."""
    import numpy as np

    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.waveform import WaveformSpec

    spec = WaveformSpec(seed=int(seed), n_subframes=int(n_subframes))
    wf = ofdm.generate(spec)
    symbols = wf.symbols.reshape(-1)
    return {"iq": ofdm.to_iq(wf.x),
            "symbols_iq": np.stack([symbols.real, symbols.imag], axis=-1).astype(np.float32),
            "metadata": {"waveform_id": spec.waveform_id, "seed": spec.seed, "n_subframes": spec.n_subframes,
                         "sample_rate_hz": spec.sample_rate_hz, "n_samples": wf.period,
                         "n_symbols": int(wf.symbols.shape[0]), "occupied_subcarriers": int(wf.symbols.shape[1]),
                         "sha256": wf.sha256()}}


def evaluate(y, *, profile: str = LTE_PROFILE, sample_rate_hz: float, nperseg: Optional[int] = None,
             bandwidth_hz: Optional[float] = None, n_sub_ch: int = 1, reference=None,
             waveform_seed: Optional[int] = None, waveform_subframes: Optional[int] = None) -> List[Dict[str, Any]]:
    """Score ``y`` with a registered metric profile; one dict per metric (name, value, unit, better, status, reason).

    ``nperseg`` is the Welch segment length of every spectral metric and has no default. The LTE profile needs the
    seed and length of the waveform that was played; without them its EVM is reported as ``missing_reference``."""
    from opendpd.core.metrics import evaluate as run_profile
    from opendpd.core.metrics.registry import get_profile
    from opendpd.schemas import SignalSpec
    from opendpd.schemas.waveform import WaveformBinding, WaveformSpec

    get_profile(profile)
    binding = None
    if (waveform_seed is None) != (waveform_subframes is None):
        raise ValueError("give both the seed and the number of subframes of the played waveform, or neither")
    if waveform_seed is not None:
        # A direct measurement has no dataset input to correlate: the profile synchronises the capture with the
        # regenerated waveform itself, and only the spec is used from the binding.
        binding = WaveformBinding(spec=WaveformSpec(seed=int(waveform_seed), n_subframes=int(waveform_subframes)),
                                  input_offset_samples=0, correlation=0.0, input_sample_rate_hz=float(sample_rate_hz))
    signal = SignalSpec(sample_rate_hz=float(sample_rate_hz), bandwidth_hz=bandwidth_hz,
                        n_sub_ch=int(n_sub_ch), nperseg=None if nperseg is None else int(nperseg), waveform=binding)
    prediction = _iq(y, "y")
    target = None if reference is None else _iq(reference, "reference")
    if target is not None and target.shape != prediction.shape:
        raise ValueError("reference must have the same number of samples as y")
    return [value.model_dump(mode="json") for value in run_profile(profile, prediction, target, signal)]
