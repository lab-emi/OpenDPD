"""Adapt the packaged dataset demodulators for read-only capture inspection.

The signal processing remains in ``datasets/<name>/demod.py`` and its shared
base classes, exactly as used by ``datasets/plot_utils.py``. This adapter binds
the dataset identity, preserves frame timing and bounds the presentation data.
It does not define EVM, fit a new reference or execute imported dataset code.
"""

import math

import numpy as np

from datasets.demodulator import Demodulator, IFFTFrameDemodulator, OFDMCPDemodulator
from opendpd.core.doctor import ANALYSIS_SAMPLES
from opendpd.core.preprocess import PREPROCESS_VERSION
from opendpd.schemas.analysis import ConstellationTrace, InspectionConstellation
from opendpd.schemas.dataset import DatasetManifest, DatasetSourceKind
from opendpd.services.datasets import analysis_window
from opendpd.services.workspace import BUILTIN_DATASETS_DIR


NOTE = (
    "Dataset-specific demodulation; no hard decisions or ideal-point snapping. "
    "Input symbols are RMS-normalized per carrier/symbol. PA output uses the "
    "input for synchronization and the existing smoothed per-subcarrier equalizer. "
    "Equalization is fitted on the displayed capture, can remove linear distortion "
    "and does not establish a standard EVM or isolate nonlinear distortion from noise."
)


def _input_origin(manifest: DatasetManifest, version: str) -> int | None:
    """Locate the selected input in raw time using recorded preprocess-v1 crops.

    Positive integer delays trim the input tail; negative delays trim its head.
    A fractional correction trims one input sample at each edge. This only
    translates recorded coordinates; the preprocessing itself is never rerun.
    """
    origin, visited = 0, set()
    while version != "raw-v1":
        if version in visited:
            return None
        visited.add(version)
        item = manifest.version(version)
        if item is None or item.code_version != PREPROCESS_VERSION or item.params is None or item.base_version is None:
            return None
        delay = item.params.delay_samples
        if not math.isfinite(delay):
            return None
        whole = math.trunc(delay)
        origin += max(-whole, 0) + int(delay != whole)
        version = item.base_version
    return origin


def dataset_constellation(manifest: DatasetManifest, version: str, x: np.ndarray, y: np.ndarray) -> InspectionConstellation:
    """Demodulate only a known packaged capture, with full contiguous samples."""
    def unavailable(reason, status="unavailable"):
        return InspectionConstellation(status=status, reason=reason, note=NOTE)

    name = manifest.source.name
    if manifest.source.kind != DatasetSourceKind.builtin or not name or not name.isidentifier():
        return unavailable("No dataset-specific demodulator is bound to this imported capture. A modulation label alone is insufficient.")
    if not (BUILTIN_DATASETS_DIR / name / "demod.py").is_file():
        return unavailable("The packaged dataset has no demodulator.")
    try:
        demod = Demodulator.from_dataset(name)
    except (OSError, ImportError, ValueError, KeyError) as exc:
        return unavailable(f"Packaged demodulator could not be loaded: {exc}")
    for field, key in (("sample_rate_hz", "input_signal_fs"), ("bandwidth_hz", "bw_main_ch"),
                       ("sub_channel_bandwidth_hz", "bw_sub_ch"), ("n_sub_ch", "n_sub_ch"),
                       ("nperseg", "nperseg"), ("modulation", "modulation")):
        if getattr(manifest.signal, field) != demod.spec.get(key):
            return unavailable(f"{field} differs from the packaged demodulator specification; its timing/carrier map cannot be assumed.")
    origin = _input_origin(manifest, version)
    if origin is None:
        return unavailable("The selected version's frame origin cannot be recovered from its preprocessing record.")
    if x.shape != y.shape:
        return unavailable("Input and output do not have matching sample coordinates.", "invalid")

    start, end = analysis_window(len(x), ANALYSIS_SAMPLES)
    if isinstance(demod, IFFTFrameDemodulator):
        nfft = demod.nperseg
        # Round inward to complete frames in RAW coordinates, including any
        # input head removed by preprocessing. Never FFT a strided sample cloud.
        start += -(origin + start) % nfft
        end -= (origin + end) % nfft
    elif isinstance(demod, OFDMCPDemodulator):
        nfft = demod.ofdm_nfft
        # The existing carrier mixer uses a zero-based clock. Cropping its
        # input head would rotate each carrier by a different phase; it has no
        # absolute-sample-offset argument. Do not silently change that receiver.
        if origin + start:
            return unavailable("The CP receiver requires the original capture start to preserve carrier phase; use raw-v1 for this plot.")
    else:
        return unavailable("The packaged demodulator does not declare a supported frame-timing strategy.")
    if end - start < nfft:
        return unavailable("The selected version does not contain a complete demodulation frame.")
    xw, yw = np.asarray(x[start:end]), np.asarray(y[start:end])
    if not np.isfinite(xw).all() or not np.isfinite(yw).all():
        return unavailable("Demodulation window contains non-finite samples.", "invalid")
    xc = xw[:, 0].astype(np.float64) + 1j * xw[:, 1].astype(np.float64)
    yc = yw[:, 0].astype(np.float64) + 1j * yw[:, 1].astype(np.float64)
    try:
        input_i, input_q = demod.demodulate(xc)
        output_i, output_q = demod.demodulate(yc, sync_signal=xc, equalize=True)
    except (ValueError, FloatingPointError, IndexError) as exc:
        return unavailable(f"Dataset demodulation failed: {exc}", "invalid")
    if not len(input_i) or not len(output_i):
        return unavailable("No complete symbols were recovered with the dataset-specific synchronization.")
    traces = []
    for label, role, ri, rq, equalized in (("Input", "input", input_i, input_q, False),
                                          ("PA output (equalized)", "primary", output_i, output_q, True)):
        if not np.isfinite(ri).all() or not np.isfinite(rq).all():
            return unavailable("The demodulator returned non-finite symbols.", "invalid")
        stride = max(1, math.ceil(len(ri) / 4000))
        traces.append(ConstellationTrace(name=label, role=role, i=ri[::stride].tolist(), q=rq[::stride].tolist(),
                                         n_symbols=len(ri), stride=stride, equalized=equalized))
    return InspectionConstellation(
        status="ok", dataset_name=name, modulation=demod.spec.get("modulation"),
        demodulator=f"datasets.{name}.demod.Demodulator ({type(demod).__bases__[0].__name__})",
        sample_range=(start, end), source_sample_range=(origin + start, origin + end),
        fft_size=nfft, active_subcarriers_per_carrier=demod.n_active, n_carriers=demod.n_sub_ch,
        traces=traces, note=NOTE,
    )
