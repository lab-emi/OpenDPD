"""``opendpd-model-v1`` packages: data only, deterministic, and equal to the evaluator.

The committed packages in ``Matlab/toolbox/tests/data`` are what the toolbox's MATLAB tests run against. The drift test
below rebuilds each from its own weights with today's PyTorch code and compares with the package's golden outputs, so a
change to a backbone fails here instead of leaving the MATLAB tests checking an outdated promise.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from opendpd.core.registry import STREAMING, streaming_variant_of
from opendpd.services import model_export as me
from opendpd.services.inference import APPLY_MODELS, OFFLINE, _offline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Matlab" / "toolbox" / "tests" / "data"
ALLOWED = {"manifest.json", "README.md", "weights.mat", "weights.npz", "golden/golden.mat", "golden/golden.npz"}
NUMERIC = {"float32", "float64", "complex128", "complex64", "int32", "int64", "int16", "int8", "uint8"}


def read_package(path):
    archive = zipfile.ZipFile(path)
    contents = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    manifest = json.loads(contents["manifest.json"])
    weights = dict(np.load(io.BytesIO(contents["weights.npz"]), allow_pickle=False))
    golden = dict(np.load(io.BytesIO(contents["golden/golden.npz"]), allow_pickle=False))
    return archive, contents, manifest, weights, golden


def fixtures():
    return sorted(DATA.glob("*.opendpd.zip")) if DATA.is_dir() else []


def test_golden_input_is_seeded_and_has_the_training_amplitude_statistics():
    a = me.golden_input(2000, rms=0.3, peak=0.8)
    b = me.golden_input(2000, rms=0.3, peak=0.8)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (2000, 2)
    magnitude = np.hypot(a[:, 0], a[:, 1])
    assert magnitude.max() <= 0.8 + 1e-6 and 0.2 < np.sqrt(np.mean(magnitude ** 2)) < 0.31
    assert not np.array_equal(a, me.golden_input(2000, rms=0.3, peak=0.8, seed=1))


def test_mat_and_npz_bytes_are_deterministic_numeric_only_and_readable():
    arrays = {"weights": np.arange(12, dtype=np.float32).reshape(3, 4), "coefficients": np.array([1 + 2j, 3 - 1j])}
    first, second = me._mat_bytes(arrays), me._mat_bytes(arrays)
    assert first == second and first[:116].rstrip() == me.MAT_HEADER
    back = loadmat(io.BytesIO(first))
    np.testing.assert_array_equal(back["weights"], arrays["weights"])
    np.testing.assert_array_equal(back["coefficients"].ravel(), arrays["coefficients"])
    one, two = me._npz_bytes(arrays), me._npz_bytes(arrays)
    assert one == two
    loaded = np.load(io.BytesIO(one), allow_pickle=False)
    assert sorted(loaded.files) == ["coefficients", "weights"]
    assert all(info.date_time == me.ZIP_TIME for info in zipfile.ZipFile(io.BytesIO(one)).infolist())


def _random_core(key, hidden=5, layers=2):
    import torch

    torch.manual_seed(3)
    if key in ("mp_ls", "gmp_ls"):
        from opendpd.core.polynomial import PolynomialModel, coefficient_count

        params = {"K": 3, "Q": 4} if key == "mp_ls" else {"Ka": 3, "La": 4, "Kb": 2, "Lb": 3, "Mb": 2, "Kc": 2, "Lc": 3, "Mc": 2}
        rng = np.random.default_rng(5)
        n = coefficient_count(key, params)
        return PolynomialModel(key, params, rng.standard_normal(n) + 1j * rng.standard_normal(n)), params
    import models as legacy

    core = legacy.CoreModel(input_size=2, hidden_size=hidden, num_layers=layers, backbone_type=key).eval()
    for tensor in core.parameters():          # reset_parameters leaves some zero biases: make every weight informative
        torch.nn.init.normal_(tensor, std=0.4)
    return core, {"hidden_size": hidden, "num_layers": layers} if key != "gmp" else {}


@pytest.mark.parametrize("key", me.EXPORT_MODELS)
def test_extract_and_inject_are_inverse_and_reproduce_the_forward_pass(key):
    import torch

    core, parameters = _random_core(key)
    arrays, architecture, sources = me.extract_weights(key, core)
    assert set(arrays) == set(sources) and all(a.dtype.name in NUMERIC for a in arrays.values())
    rebuilt = me.core_from_weights(key, parameters, architecture, arrays)
    x = torch.from_numpy((0.3 * np.random.default_rng(1).standard_normal((2, 64, 2))).astype(np.float32))
    with torch.inference_mode():
        np.testing.assert_allclose(rebuilt(x).numpy(), core(x).numpy(), rtol=0, atol=1e-6)
    assert not np.allclose(core(x).detach().numpy(), 0)


def test_the_supported_set_is_the_set_apply_is_tested_for():
    assert me.EXPORT_MODELS == APPLY_MODELS == ("gru", "tres_gru", "gmp", "mp_ls", "gmp_ls")


@pytest.mark.skipif(not fixtures(), reason="no committed model packages in this checkout")
@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.name.replace(".opendpd.zip", ""))
def test_committed_package_is_data_only_and_self_consistent(path):
    archive, contents, manifest, weights, golden = read_package(path)
    assert set(contents) <= ALLOWED and {"manifest.json", "weights.mat", "golden/golden.mat"} <= set(contents)
    assert not any(name.endswith((".m", ".py", ".so", ".dll", ".p", ".mex", ".exe", ".sh")) for name in contents)
    assert manifest["format"] == me.FORMAT and manifest["model"]["key"] in me.EXPORT_MODELS
    for name, digest in manifest["files"].items():
        assert hashlib.sha256(contents[name]).hexdigest() == digest, name
    assert set(manifest["files"]) == set(contents) - {"manifest.json"}
    mat = loadmat(io.BytesIO(contents["weights.mat"]))
    assert {k for k in mat if not k.startswith("__")} == set(weights)
    assert {w["name"] for w in manifest["model"]["weights"]} == set(weights)
    for entry in manifest["model"]["weights"]:
        assert list(weights[entry["name"]].shape) == entry["shape"] and str(weights[entry["name"]].dtype) == entry["dtype"]
    assert manifest["golden"]["tolerance_abs"] == me.TOLERANCE_ABS
    key = manifest["model"]["key"]
    assert manifest["execution"][STREAMING]["available"] == (streaming_variant_of(key) is not None)
    assert manifest["execution"][OFFLINE]["segment_samples"] == manifest["signal"]["nperseg"]
    assert "input_description" in manifest["golden"] and "not user data" in manifest["golden"]["input_description"]
    assert golden["input"].dtype == np.float32 and golden["input"].shape == (manifest["golden"]["samples"], 2)


@pytest.mark.skipif(not fixtures(), reason="no committed model packages in this checkout")
@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.name.replace(".opendpd.zip", ""))
def test_committed_package_still_matches_todays_pytorch_code(path):
    """Rebuild the module from the package's own weights and compare with the golden outputs it ships."""
    from opendpd.services.streaming import stream_outputs

    _, _, manifest, weights, golden = read_package(path)
    key = manifest["model"]["key"]
    core = me.core_from_weights(key, manifest["model"]["parameters"], manifest["model"]["architecture"], weights)
    x = golden["input"]
    offline = _offline(core, x, manifest["signal"]["nperseg"], batch_segments=16)
    assert np.max(np.abs(offline - golden["output_offline_segmented"])) <= me.TOLERANCE_ABS
    assert np.max(np.abs(offline - x)) > 1e-3                         # not an identity in disguise
    variant = streaming_variant_of(key)
    if variant is None:
        assert "output_streaming_stateful" not in golden
    else:
        chunk = manifest["golden"]["streaming_chunk_samples"]
        streamed, _ = stream_outputs(core.cpu(), variant.key, x, chunk_samples=chunk,
                                     sample_rate_hz=manifest["signal"]["sample_rate_hz"])
        assert np.max(np.abs(streamed - golden["output_streaming_stateful"])) <= me.TOLERANCE_ABS
        assert np.max(np.abs(streamed - offline)) > 1e-6              # the two semantics differ, which is why both are shipped
