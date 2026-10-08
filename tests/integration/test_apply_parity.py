"""apply() reproduces the evaluator for every supported model and honours the streaming contract.

Reference for the offline semantics is ``predict_test_split`` (the code that scored the run); reference for the
streaming semantics is ``stream_outputs`` over the same rebuilt network. Both are independent of the code under test
in the sense that matters here: apply goes through the SDK's isolated process and its own segmentation.
"""

import json

import numpy as np
import pytest

from opendpd.core.registry import streaming_variant_of
from opendpd.sdk import SDKError, open_project
from opendpd.services import experiments
from opendpd.services.evaluation import predict_test_split, trained_model
from opendpd.services.inference import APPLY_MODELS
from opendpd.services.streaming import stream_outputs
from opendpd.services.workspace import Workspace
from tests.fixtures.synthetic import synthesize

pytestmark = pytest.mark.integration
NPERSEG = 128
FS = 80e6
TRAINING = {"epochs": 1, "frame_length": 32, "frame_stride": 32, "batch_size": 16, "batch_size_eval": 16}
# small enough to train in a second or two on CPU, large enough to be non-trivial
PARAMETERS = {
    "gru": {"hidden_size": 6},
    "tres_gru": {"hidden_size": 6},
    "gmp": {},
    "mp_ls": {"K": 3, "Q": 4},
    "gmp_ls": {"Ka": 3, "La": 4, "Kb": 2, "Lb": 3, "Mb": 1, "Kc": 2, "Lc": 3, "Mc": 1, "rcond": 1e-6},
}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    p = open_project(tmp_path_factory.mktemp("apply-parity") / "workspace")
    yield p
    p.close(stop_service=True)


@pytest.fixture(scope="module")
def dataset(project):
    x, y = synthesize(4096, fs=FS, bandwidth=20e6)
    return project.import_iq(x, y, dataset_id="parity", sample_rate_hz=FS, bandwidth_hz=20e6, nperseg=NPERSEG,
                             origin="synthetic")


@pytest.fixture(scope="module")
def surrogate(project, dataset):
    return project.train_pa(dataset["dataset_id"], parameters=PARAMETERS["gru"], training=TRAINING,
                            device="cpu").wait(timeout=180)


@pytest.fixture(scope="module")
def runs(project, dataset, surrogate):
    """model key -> (PA run, DPD run). Every DPD cascades with the same gradient-trained GRU surrogate."""
    out = {}
    for key in APPLY_MODELS:
        pa = surrogate if key == "gru" else project.train_pa(
            dataset["dataset_id"], model=key, parameters=PARAMETERS[key], training=TRAINING, device="cpu").wait(timeout=180)
        dpd = project.train_dpd(dataset["dataset_id"], surrogate, model=key, parameters=PARAMETERS[key],
                                training=TRAINING, device="cpu").wait(timeout=180)
        out[key] = (pa, dpd)
    return out


@pytest.fixture(scope="module")
def waveform(project, dataset):
    from modules.data_collector import load_dataset

    ws = Workspace.open(project.workspace)
    return load_dataset(dataset_path=ws.dataset_version_dir(dataset["dataset_id"], "raw-v1"))[4]


@pytest.mark.parametrize("role", [0, 1], ids=["pa", "dpd"])
@pytest.mark.parametrize("key", APPLY_MODELS)
def test_offline_apply_matches_the_evaluator(project, runs, waveform, key, role):
    job = runs[key][role]
    ws = Workspace.open(project.workspace)
    expected = predict_test_split(ws, job.run_id, experiments.load_resolved(ws, job.run_id),
                                  experiments.load_artifacts(ws, job.run_id))
    expected_iq = (expected.prediction if role == 0 else expected.u).reshape(-1, 2)[:len(waveform)]
    actual, meta = job.apply(waveform)
    np.testing.assert_allclose(actual, expected_iq, rtol=1e-5, atol=1e-6)
    assert actual.shape == waveform.shape and actual.dtype == np.float32
    assert meta["execution"] == "offline_segmented" and meta["segment_samples"] == NPERSEG
    assert meta["model"] == key and meta["sample_rate_hz"] == FS
    assert meta["output_role"] == ("modeled_pa_output" if role == 0 else "predistorted_pa_input")
    assert meta["checkpoint_sha256"] and len(meta["output_sha256"]) == 64


@pytest.mark.parametrize("key", APPLY_MODELS)
def test_offline_segments_are_independent_and_the_tail_is_trimmed(runs, key):
    job = runs[key][1]
    x, _ = synthesize(2 * NPERSEG + 5, fs=FS, bandwidth=20e6)
    whole, _ = job.apply(x)
    parts = [job.apply(x[:NPERSEG])[0], job.apply(x[NPERSEG:2 * NPERSEG])[0], job.apply(x[2 * NPERSEG:])[0]]
    np.testing.assert_allclose(whole, np.concatenate(parts), rtol=1e-5, atol=1e-6)
    assert whole.shape == x.shape


def test_non_causal_models_state_what_they_read_beyond_a_segment(runs):
    x, _ = synthesize(NPERSEG, fs=FS, bandwidth=20e6)
    _, tres = runs["tres_gru"][1].apply(x)
    _, causal = runs["gru"][1].apply(x)
    assert tres["lookahead_samples"] == 16 and any("16 future samples" in note for note in tres["limitations"])
    assert causal["lookahead_samples"] == 0 and not any("future samples" in note for note in causal["limitations"])


@pytest.mark.parametrize("role", [0, 1], ids=["pa", "dpd"])
@pytest.mark.parametrize("key", ["gru", "gmp"])
def test_streaming_apply_follows_the_streaming_contract(project, runs, waveform, key, role):
    job = runs[key][role]
    variant = streaming_variant_of(key)
    streamed, meta = job.apply(waveform, execution="streaming", chunk_samples=300)
    ws = Workspace.open(project.workspace)
    core = trained_model(ws, job.run_id).evaluated
    reference, _ = stream_outputs(core.cpu(), variant.key, waveform, chunk_samples=300, sample_rate_hz=FS)
    np.testing.assert_allclose(streamed, reference, rtol=1e-5, atol=1e-6)
    assert meta["execution"] == "streaming_stateful" and meta["streaming_variant"] == variant.key
    assert meta["streaming"]["chunk_samples"] == 300 and meta["streaming"]["consistency"]["within_tolerance"]
    assert "segment_samples" not in meta and "only at the start" in meta["state_reset"]
    # chunking must not change the signal beyond the contract's tolerance, and the state must really be carried
    other, _ = job.apply(waveform, execution="streaming", chunk_samples=77)
    warmup = meta["streaming"]["warmup_samples"]
    np.testing.assert_allclose(streamed[warmup:], other[warmup:], atol=1e-4)
    offline, _ = job.apply(waveform)
    assert np.abs(streamed[NPERSEG:] - offline[NPERSEG:]).max() > 1e-6


def test_streaming_alias_is_accepted(runs, waveform):
    named, _ = runs["gru"][1].apply(waveform[:300], execution="streaming_stateful")
    alias, _ = runs["gru"][1].apply(waveform[:300], execution="streaming")
    np.testing.assert_array_equal(named, alias)


@pytest.mark.parametrize("key", ["tres_gru", "mp_ls", "gmp_ls"])
def test_models_without_a_streaming_variant_are_refused_not_approximated(runs, waveform, key):
    with pytest.raises(SDKError, match="no_streaming_variant.*gmp, gru"):
        runs[key][1].apply(waveform[:300], execution="streaming")


def test_invalid_options_and_unsupported_models_are_refused(project, dataset, runs, waveform):
    job = runs["gru"][1]
    with pytest.raises(ValueError, match="execution must be one of"):
        job.apply(waveform, execution="realtime")
    with pytest.raises(ValueError, match="chunk_samples"):
        job.apply(waveform, execution="streaming", chunk_samples=0)
    lstm = project.train_pa(dataset["dataset_id"], model="lstm", parameters={"hidden_size": 4}, training=TRAINING,
                            device="cpu").wait(timeout=180)
    with pytest.raises(SDKError, match="unsupported_model"):
        lstm.apply(waveform[:300])
    with pytest.raises(ValueError, match=r"shape \(N, 2\)|I/Q"):
        job.apply(np.zeros((0, 2), np.float32))


def test_metadata_is_json_serialisable_and_names_its_evidence(runs, waveform):
    _, meta = runs["mp_ls"][1].apply(waveform[:300])
    json.dumps(meta, allow_nan=False)
    assert meta["evidence"].startswith("model inference") and meta["apply_version"] == 2
