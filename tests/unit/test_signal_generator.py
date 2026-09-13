"""Numerical oracles for synthetic signals, not RF conformance evidence."""
import io
import zipfile

import numpy as np
import pytest
from pydantic import ValidationError

from opendpd.core.waveforms.generator import allocation, synthesize
from opendpd.core.waveforms.generator_presets import presets, coverage
from opendpd.schemas.signal_generator import GeneratorConfig, GeneratorDatasetRequest
from opendpd.services.signal_generator import create_dataset, export_signal, generate, read_signal, sample_counts
from opendpd.services.workspace import Workspace, WorkspaceError
from opendpd.services.datasets import load_version_arrays


def preset(identifier, **changes):
    source = next(p.config for p in presets() if p.preset_id == identifier).model_dump()
    return GeneratorConfig.model_validate({**source, **changes})


@pytest.mark.parametrize("item", presets(), ids=lambda p: p.preset_id)
def test_presets_are_finite_exact_length_and_explicit_about_coverage(item):
    x, a = synthesize(item.config)
    assert x.dtype == np.complex64 and len(x) == item.config.sample_count
    assert np.isfinite(x).all() and np.isfinite(a.psd_dbfs_hz).all()
    assert a.rms == pytest.approx(item.config.rms, rel=1e-6)
    assert a.papr_db == pytest.approx(10*np.log10(np.max(np.abs(x.astype(complex))**2)/np.mean(np.abs(x.astype(complex))**2)))
    assert a.ccdf_probability == sorted(a.ccdf_probability, reverse=True)
    assert a.duration_ms == len(x)/item.config.sample_rate_hz*1000
    if item.config.waveform == "ofdm":
        assert a.evm_percent < .0001
        assert a.pilot_carriers + a.data_carriers == sum(item.config.channel_subcarriers)
        assert any("Not a conformance" in note for note in a.notes)
    if item.family == "wifi8":
        assert coverage(item.config) == "experimental"
        assert any("UHR" in note for note in a.notes)


def test_one_ms_nr_prefix_lengths_follow_subframe_timing():
    config = preset("nr-20", length_mode="duration", duration_ms=1)
    x, a = synthesize(config)
    assert len(x) == 122880 and a.complete_symbols == 28 and a.trailing_samples == 0
    assert a.cp_lengths_samples == [352] + [288]*13 + [352] + [288]*13
    # Cyclic prefixes must be actual copies of the corresponding useful symbol tail.
    cursor = 0
    for cp in a.cp_lengths_samples:
        np.testing.assert_array_equal(x[cursor:cursor+cp], x[cursor+4096:cursor+4096+cp])
        cursor += cp+4096


def test_tone_has_analytic_zero_db_papr_and_rf_metadata_does_not_mix():
    config = preset("custom-tone", n_samples=4096)
    x, a = synthesize(config)
    time = np.arange(4096)/config.sample_rate_hz
    np.testing.assert_allclose(x, config.rms*np.exp(2j*np.pi*config.tone_frequency_hz*time), atol=2e-8)
    assert abs(a.papr_db) < 1e-6
    changed = config.model_copy(update={"carrier_frequency_hz": 28e9})
    np.testing.assert_array_equal(x, synthesize(changed)[0])


def test_pilots_channel_powers_and_nulls_are_real_fft_allocations():
    config = GeneratorConfig(fft_size=512, sample_rate_hz=80e6, bandwidth_hz=20e6,
        channel_subcarriers=[48, 72], channel_modulations=[4, 16], channel_power_db=[0, -6],
        channel_gap_bins=4, pilot_mode="explicit", pilot_indices=[-50, -25, 25, 50], n_samples=32768)
    channels, pilots = allocation(config)
    x, a = synthesize(config)
    assert len(channels[0]) == 48 and len(channels[1]) == 72 and len(pilots) == 4
    assert not set(channels[0]) & set(channels[1]) and 0 not in np.concatenate(channels)
    spectrum = np.fft.fft(x[64:64+2048])
    empty = np.setdiff1d(np.arange(2048), np.concatenate(channels) % 2048)
    assert np.max(np.abs(spectrum[empty])) < 1e-5
    assert a.evm_percent < .0001


def test_impairments_are_visible_and_seed_is_reproducible():
    clean = preset("wifi6-20", n_samples=65536)
    noisy = preset("wifi6-20", n_samples=65536, snr_db=20, iq_gain_db=1, frequency_offset_hz=500)
    x, a = synthesize(noisy)
    np.testing.assert_array_equal(x, synthesize(noisy)[0])
    assert a.evm_percent > 10
    assert not np.array_equal(x, synthesize(noisy.model_copy(update={"seed": 43}))[0])
    clipped = clean.model_copy(update={"clip_db": 3})
    z, _ = synthesize(clipped)
    assert np.max(np.abs(z)) <= clean.rms*10**(3/20)*(1+1e-6)


@pytest.mark.parametrize("changes", [
    {"sample_rate_hz": float("inf")}, {"fft_size": 1000}, {"channel_subcarriers": [512]},
    {"channel_subcarriers": [100, 100]}, {"bandwidth_hz": 1e6}, {"cp_samples": 512},
    {"cp_mode": "nr_normal"}, {"pilot_indices": [1, 1]}, {"channel_power_db": [float("nan")]},
    {"length_mode": "duration", "duration_ms": 1000}, {"frequency_offset_hz": 40e6},
])
def test_invalid_geometry_and_unbounded_work_are_rejected(changes):
    with pytest.raises(ValidationError):
        GeneratorConfig(**changes)


def test_explicit_pilots_must_be_on_active_carriers_and_leave_payload():
    config = GeneratorConfig(pilot_mode="explicit", pilot_indices=[1000])
    with pytest.raises(ValueError, match="active"):
        synthesize(config)


def test_native_wlan_rate_accepts_nominal_bandwidth_but_checks_asymmetric_bins():
    config = preset("wifi6-20", oversampling=1, sample_rate_hz=20e6)
    assert synthesize(config)[1].subcarrier_spacing_hz == 78125
    with pytest.raises(ValidationError, match="declared baseband bandwidth"):
        GeneratorConfig(channel_subcarriers=[3], bandwidth_hz=4*78125)


def test_private_export_and_training_dataset_preserve_exact_input(tmp_path):
    ws = Workspace.create(tmp_path / "workspace")
    result = generate(ws, preset("wifi7-20", n_samples=16384))
    assert generate(ws, result.config).signal_id == result.signal_id
    with zipfile.ZipFile(export_signal(ws, result.signal_id)) as package:
        saved = np.load(io.BytesIO(package.read("iq.npy")), allow_pickle=False)
        csv = np.loadtxt(io.StringIO(package.read("iq.csv").decode()), delimiter=",", skiprows=1, dtype=np.float32)
        np.testing.assert_array_equal(saved, csv)
        assert len(csv) == 16384 and b"SYNTHETIC" in package.read("README.txt")
    request = GeneratorDatasetRequest(dataset_id="generated-pa")
    response = create_dataset(ws, result.signal_id, request)
    assert response.dataset.origin.value == "synthetic" and response.dataset.simulation["physical_measurement"] is False
    x, y, split = load_version_arrays(ws, "generated-pa")
    np.testing.assert_array_equal(x, saved)
    assert not np.array_equal(x, y)
    assert response.test_samples == split.boundaries["test"][1]-split.boundaries["test"][0]
    assert sample_counts(ws, "generated-pa", "raw-v1").counts["test"] == response.test_samples
    assert create_dataset(ws, result.signal_id, request).dataset.raw_sha256 == response.dataset.raw_sha256
    with pytest.raises(WorkspaceError, match="different settings"):
        create_dataset(ws, result.signal_id, request.model_copy(update={"compression": .8}))
    with pytest.raises(WorkspaceError, match="does not exist"):
        sample_counts(ws, "generated-pa", "missing-v1")
    (ws.root / "signals" / result.signal_id / "iq.npy").write_bytes(b"changed")
    with pytest.raises(WorkspaceError, match="changed"):
        read_signal(ws, result.signal_id)


def test_test_counts_use_selected_version_boundaries(tmp_path):
    from opendpd.schemas.dataset import DatasetVersion
    ws = Workspace.create(tmp_path / "workspace")
    result = generate(ws, preset("custom-ofdm", n_samples=16384))
    ds = create_dataset(ws, result.signal_id, GeneratorDatasetRequest(dataset_id="counts")).dataset
    split = ds.split.model_copy(update={"boundaries": {"train": (0, 7500), "val": (7756, 9000), "test": (9256, 12500)}})
    ws.save_dataset(ds.model_copy(update={"versions": [*ds.versions, DatasetVersion(version="cropped-v1", n_samples=12500, split=split)]}))
    assert sample_counts(ws, "counts", "cropped-v1").counts["test"] == 3244
    assert sample_counts(ws, "counts", "raw-v1").counts["test"] != 3244
