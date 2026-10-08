"""Export a finished run as an ``opendpd-model-v1`` package: weights and a golden test vector, no executable code.

A package is a zip with ``manifest.json`` (model, execution semantics, signal metadata, scaling note, evidence type,
provenance and the SHA-256 of every other file), ``weights.mat`` / ``weights.npz`` (numeric arrays only, the same
names in both), ``golden/golden.mat`` / ``golden/golden.npz`` (an input and the outputs ``apply`` produced for it) and
a README. The OpenDPD toolbox for MATLAB runs it with plain MATLAB code (``opendpd.load``); nothing in the package is
ever executed. A model is exportable only together with a golden test against the evaluator
(``EXPORT_MODELS`` equals the set ``apply`` is tested for), and the export is deterministic: the same run gives the
same bytes.

The golden outputs are ``apply_waveform``'s, so they are the evaluator's by construction (see
``opendpd/services/inference.py``). The input is synthetic (seeded noise with the amplitude statistics of the
training input), never a slice of the user's data: a package can be shared without sharing a measurement.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from opendpd.core.registry import STREAMING, get_model, streaming_variant_of
from opendpd.schemas import RunStatus, TaskType
from opendpd.services.inference import APPLY_MODELS, OFFLINE, InferenceError
from opendpd.services.workspace import Workspace

FORMAT = "opendpd-model-v1"
EXPORT_MODELS = APPLY_MODELS
TOLERANCE_ABS = 1e-5                   # float32 outputs: the acceptance bound of the golden test
GOLDEN_SEED = 20260101
GOLDEN_CHUNK_SAMPLES = 37              # an awkward size on purpose: chunk boundaries must not matter
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
GMP_STREAM_HISTORY_NOTE = "output n reads x[n-2*(memory_length-1)..n]; that history is carried instead of zero-filled"

README = """# OpenDPD model package ({key}, {role})

Format: `{format}`. This package holds data only: `manifest.json`, `weights.mat` and `weights.npz` (the same arrays),
`golden/` (a test input and the outputs OpenDPD produced for it) and this file. Nothing in it is executed.

* In MATLAB, with the OpenDPD toolbox: `model = opendpd.load("<this file>")`, then `opendpd.verify(model)` checks the
  golden outputs (absolute error {tolerance:g}), `y = opendpd.apply(model, x)` applies the model. No Python is needed.
* Anywhere else: read `weights.npz` and `manifest.json`; `manifest.json` states the architecture and the execution
  semantics, and `golden/golden.npz` is the test vector.

What it is: the {role} model `{key}` from run `{run_id}`. Evidence: {evidence}. Samples are not normalised or
aligned: supply them in the units and at the sample rate of the training dataset ({sample_rate_hz:g} Hz,
amplitude units: {amplitude_units}). Input amplitude seen in training: rms {rms:.6g}, peak {peak:.6g}; outputs
for inputs far outside that range are extrapolation.

Execution semantics are in `manifest.json` under `execution`: `offline_segmented` is how the run was scored
(state reset every {nperseg} samples); `streaming_stateful` is {streaming}.
Licence: Apache-2.0, as OpenDPD.
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_weights(key: str, core: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, str]]:
    """``(arrays, architecture, source names)`` of the module under evaluation; the arrays keep the checkpoint's dtype."""
    if key in ("mp_ls", "gmp_ls"):
        w = core.coefficients.detach().cpu().numpy().astype(np.complex128).reshape(-1)
        order = ("k * Q + q, lag fastest: w[k*Q+q] multiplies x(n-q)*|x(n-q)|^k" if key == "mp_ls"
                 else "aligned (k*La+l), then lagging (k,l,m), then leading (k,l,m) terms, as opendpd.core.polynomial.gmp_basis")
        return ({"coefficients": w}, {"parameters": {k: v for k, v in core.params.items()}, "coefficient_order": order},
                {"coefficients": "coefficients"})
    backbone = core.backbone
    state = {name: tensor.detach().cpu().numpy() for name, tensor in backbone.state_dict().items()}
    if key == "gmp":
        weight = state["Weight"].astype(np.float32).reshape(-1)
        return ({"gmp_weight": weight}, {"memory_length": int(backbone.memory_length), "degree": int(backbone.degree),
                                         "weights": "real", "terms": weight.size}, {"gmp_weight": "Weight"})
    arrays: Dict[str, np.ndarray] = {}
    names: Dict[str, str] = {}
    layers = int(backbone.num_layers)
    for layer in range(layers):
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            source = f"rnn.{kind}_l{layer}"
            if source in state:
                arrays[f"rnn_{kind}_l{layer}"] = state[source].astype(np.float32)
                names[f"rnn_{kind}_l{layer}"] = source
    arrays["fc_weight"], names["fc_weight"] = state["fc_out.weight"].astype(np.float32), "fc_out.weight"
    if "fc_out.bias" in state:
        arrays["fc_bias"], names["fc_bias"] = state["fc_out.bias"].astype(np.float32), "fc_out.bias"
    architecture: Dict[str, Any] = {"hidden_size": int(backbone.hidden_size), "num_layers": layers,
                                    "rnn_bias": "rnn.bias_ih_l0" in state, "output_bias": "fc_out.bias" in state}
    if key == "gru":
        architecture.update(input_features=["I", "Q"], gru="torch.nn.GRU gate order r, z, n; n = tanh(W_in x + b_in + r*(W_hn h + b_hn))",
                            head="linear(hidden -> 2)")
    else:
        arrays["tcn_conv1_weight"], names["tcn_conv1_weight"] = state["tcn.0.weight"].astype(np.float32), "tcn.0.weight"
        arrays["tcn_conv2_weight"], names["tcn_conv2_weight"] = state["tcn.2.weight"].astype(np.float32), "tcn.2.weight"
        architecture.update(
            input_features=["I", "Q", "|x|", "|x|^3", "I(n+1)", "Q(n+1)"],
            next_sample="I(n+1), Q(n+1) are the next sample of the segment; the last sample of a segment reads the first (torch.roll)",
            tcn="conv1d(2->3, kernel 3, dilation 16, zero padding 16) -> hardswish -> conv1d(3->2, kernel 1) -> hardswish; "
                "reads x(n-16), x(n), x(n+16)",
            head="linear(hidden -> 2) + tcn output")
    return arrays, architecture, names


def core_from_weights(key: str, parameters: Dict[str, Any], architecture: Dict[str, Any], arrays: Dict[str, np.ndarray]):
    """Rebuild the module that ``extract_weights`` read (the inverse, used by tests and by importers)."""
    import torch

    if key in ("mp_ls", "gmp_ls"):
        from opendpd.core.polynomial import PolynomialModel

        return PolynomialModel(key, parameters, arrays["coefficients"])
    import models as legacy

    core = legacy.CoreModel(input_size=2, hidden_size=int(architecture.get("hidden_size", 1)),
                            num_layers=int(architecture.get("num_layers", 1)), backbone_type=key)
    backbone = core.backbone
    state: Dict[str, Any] = {}
    if key == "gmp":
        state["Weight"] = torch.from_numpy(np.asarray(arrays["gmp_weight"], dtype=np.float32).reshape(1, -1))
    else:
        for name, values in arrays.items():
            if name.startswith("rnn_"):
                state["rnn." + name[len("rnn_"):]] = torch.from_numpy(np.asarray(values, dtype=np.float32))
        state["fc_out.weight"] = torch.from_numpy(np.asarray(arrays["fc_weight"], dtype=np.float32))
        if "fc_bias" in arrays:
            state["fc_out.bias"] = torch.from_numpy(np.asarray(arrays["fc_bias"], dtype=np.float32))
        if key == "tres_gru":
            state["tcn.0.weight"] = torch.from_numpy(np.asarray(arrays["tcn_conv1_weight"], dtype=np.float32))
            state["tcn.2.weight"] = torch.from_numpy(np.asarray(arrays["tcn_conv2_weight"], dtype=np.float32))
    backbone.load_state_dict(state)
    return core.eval()


def golden_input(n: int, rms: float, peak: float, seed: int = GOLDEN_SEED) -> np.ndarray:
    """Seeded, mildly band-limited complex noise with the training input's rms, clipped to its peak; float32 ``(n, 2)``."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    z = np.convolve(z, np.ones(5) / 5.0, mode="same")
    z *= rms / np.sqrt(np.mean(np.abs(z) ** 2))
    magnitude = np.abs(z)
    over = magnitude > peak
    z[over] *= peak / magnitude[over]
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


MAT_HEADER = b"MATLAB 5.0 MAT-file, written by OpenDPD (opendpd-model-v1); numeric arrays only"


def _mat_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    """A MAT v5 file of numeric arrays. SciPy puts the creation time into the 116-byte text header; it is replaced by
    a fixed text so that the same run gives the same bytes."""
    from scipy.io import savemat

    buffer = io.BytesIO()
    savemat(buffer, arrays, format="5", do_compression=True, oned_as="column")
    data = bytearray(buffer.getvalue())
    data[:116] = MAT_HEADER.ljust(116, b" ")
    return bytes(data)


def _npz_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    """``numpy.savez_compressed``-compatible, with fixed timestamps (NumPy stamps the current time into the zip)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def _train_input_stats(ws: Workspace, resolved: Any, dataset: Any) -> Dict[str, float]:
    from modules.data_collector import load_dataset

    x_train = load_dataset(dataset_path=ws.dataset_version_dir(dataset.dataset_id, resolved.dataset.preprocessing_version))[0]
    amplitude = np.hypot(np.asarray(x_train, dtype=np.float64)[:, 0], np.asarray(x_train, dtype=np.float64)[:, 1])
    return {"rms": float(np.sqrt(np.mean(amplitude ** 2))), "peak": float(amplitude.max()), "n_samples": int(amplitude.size)}


def _execution(key: str, nperseg: int) -> Dict[str, Any]:
    model = get_model(key)
    execution: Dict[str, Any] = {
        OFFLINE: {"segment_samples": nperseg, "lookahead_samples": model.lookahead_samples,
                  "state_reset": "zero at the start of every segment and of every apply call",
                  "tail": "the last segment is zero padded to segment_samples; the padding is trimmed from the output",
                  "lookahead_note": model.lookahead_note}}
    variant = streaming_variant_of(key)
    if variant is None:
        execution[STREAMING] = {"available": False,
                                "reason": f"'{key}' has no registered streaming variant; use {OFFLINE}, which is how the run was scored"}
    else:
        windowed = variant.key == "gmp_stream"
        execution[STREAMING] = {"available": True, "variant": variant.key, "state": "window" if windowed else "recurrent",
                                "lookahead_samples": 0, "history_samples": None,
                                "state_reset": "only at the start of a stream; chunk boundaries never reset it",
                                "note": GMP_STREAM_HISTORY_NOTE if windowed else "hidden state carried across chunks"}
    return execution


def export_model(ws: Workspace, run_id: str, destination: Any, *, golden_samples: Optional[int] = None) -> Dict[str, Any]:
    """Write ``destination`` (a ``.zip`` file path) for a succeeded run; returns a summary of what was written."""
    from opendpd.services.evaluation import trained_model
    from opendpd.services.experiments import load_resolved, load_result, load_run
    from opendpd.services.inference import _checkpoint, apply_waveform

    run = load_run(ws, run_id)
    resolved = load_resolved(ws, run_id)
    if run.status != RunStatus.succeeded or resolved.task not in (TaskType.train_pa, TaskType.train_dpd):
        raise InferenceError("model_unavailable", "Export needs a succeeded train_pa or train_dpd run")
    key = resolved.model.key
    if key not in EXPORT_MODELS:
        raise InferenceError("unsupported_model", f"export supports {', '.join(EXPORT_MODELS)}; '{key}' has no golden "
                             "test against the evaluator yet")
    if resolved.quantization and resolved.quantization.enabled:
        raise InferenceError("unsupported_model", "Quantisation-aware runs are not supported by export")
    artifact = _checkpoint(ws, run_id)
    result = load_result(ws, run_id)
    if result is None or result.nperseg is None or result.evaluated_signal is None:
        raise InferenceError("metadata_missing", "Run needs a stored result with frozen signal and segment metadata")
    nperseg = int(result.nperseg)
    trained = trained_model(ws, run_id)
    arrays, architecture, sources = extract_weights(key, trained.evaluated)
    stats = _train_input_stats(ws, resolved, trained.dataset)
    role = "dpd" if run.task == TaskType.train_dpd else "pa"

    n = int(golden_samples or min(max(2 * nperseg + nperseg // 2, 256), 8192))
    x = golden_input(n, stats["rms"], stats["peak"])
    golden: Dict[str, np.ndarray] = {"input": x}
    y_offline, _ = apply_waveform(ws, run_id, x)
    golden["output_offline_segmented"] = y_offline
    streaming = streaming_variant_of(key)
    if streaming is not None:
        y_stream, meta = apply_waveform(ws, run_id, x, execution=STREAMING, chunk_samples=GOLDEN_CHUNK_SAMPLES)
        golden["output_streaming_stateful"] = y_stream
        history = meta.get("streaming", {}).get("history_samples")
    else:
        history = None

    execution = _execution(key, nperseg)
    if streaming is not None:
        execution[STREAMING]["history_samples"] = history
    evidence_type = {"synthetic": "simulation", "measured": "measurement"}.get(str(getattr(trained.dataset.origin, "value",
                                                                                   trained.dataset.origin)), "unknown")
    files: Dict[str, bytes] = {
        "weights.mat": _mat_bytes(arrays), "weights.npz": _npz_bytes(arrays),
        "golden/golden.mat": _mat_bytes(golden), "golden/golden.npz": _npz_bytes(golden),
    }
    signal = result.evaluated_signal
    manifest: Dict[str, Any] = {
        "format": FORMAT,
        "created_by": {"opendpd": _version(), "exporter": "opendpd.services.model_export"},
        "run": {"run_id": run_id, "role": role, "task": run.task.value, "checkpoint_sha256": artifact.file.sha256,
                "dataset_id": trained.dataset.dataset_id},
        "model": {"key": key, "parameters": resolved.model.parameters, "architecture": architecture,
                  "weights": [{"name": name, "shape": list(values.shape), "dtype": str(values.dtype), "source": sources[name]}
                              for name, values in arrays.items()]},
        "signal": {"sample_rate_hz": signal.sample_rate_hz, "bandwidth_hz": signal.bandwidth_hz,
                   "nperseg": nperseg, "amplitude_units": signal.amplitude_units},
        "scaling": {"reference_gain": result.scaling.reference_gain if result.scaling else None,
                    "train_input": stats,
                    "note": "No normalisation, alignment or gain fitting is applied: supply samples in the training dataset's "
                            "units and sample rate. A DPD output is the predistorted PA input."},
        "execution": execution,
        "evidence": {"type": evidence_type, "dataset_origin": str(getattr(trained.dataset.origin, "value", trained.dataset.origin)),
                     "note": "model inference; no hardware measurement or new metric evaluation"},
        "golden": {"input": "golden/golden.mat", "samples": n, "tolerance_abs": TOLERANCE_ABS,
                   "streaming_chunk_samples": GOLDEN_CHUNK_SAMPLES if streaming is not None else None,
                   "input_description": f"seeded noise ({GOLDEN_SEED}), rms and peak of the training input; not user data",
                   "outputs": sorted(k for k in golden if k != "input")},
    }
    readme = README.format(key=key, role=role, format=FORMAT, tolerance=TOLERANCE_ABS,
                           run_id=run_id, evidence=f"{evidence_type} (dataset origin: {manifest['evidence']['dataset_origin']})",
                           sample_rate_hz=signal.sample_rate_hz, amplitude_units=signal.amplitude_units, rms=stats["rms"],
                           peak=stats["peak"], nperseg=nperseg,
                           streaming=("available as " + streaming.key) if streaming is not None else "not available for this model")
    files["README.md"] = readme.encode("utf-8")
    manifest["files"] = {name: _sha256(data) for name, data in sorted(files.items())}
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    partial.replace(destination)
    return {"path": str(destination), "sha256": _sha256(destination.read_bytes()), "format": FORMAT, "model": key, "role": role,
            "run_id": run_id, "files": sorted(files), "golden_samples": n,
            "execution": {k: bool(v.get("available", True)) for k, v in execution.items()}}


def _version() -> str:
    from opendpd import __version__

    return str(__version__)
