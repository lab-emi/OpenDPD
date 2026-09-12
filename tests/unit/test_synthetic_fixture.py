"""The synthetic fixture generator must produce what its manifest promises."""

import json
from pathlib import Path

import numpy as np

from tests.fixtures.synthetic import TIERS, Impairments, synthesize, write_dataset

MANIFEST = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "manifest.json").read_text())


def test_tiny_tier_matches_manifest():
    assert TIERS["tiny"] == MANIFEST["tiers"]["tiny"]["synthetic"]["samples"]


def test_synthesize_is_deterministic_and_finite():
    x1, y1 = synthesize(4096, seed=7)
    x2, y2 = synthesize(4096, seed=7)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert x1.shape == y1.shape == (4096, 2)
    assert x1.dtype == y1.dtype == np.float32
    assert np.isfinite(x1).all() and np.isfinite(y1).all()


def test_known_impairments_are_applied():
    imp = Impairments(delay_samples=5, gain=2.0 + 0j, clip_level=0.6, n_outliers=3, nan_indices=(10,))
    x, y = synthesize(8192, seed=1, impairments=imp)
    # integer delay: the first `delay` output samples are zero-filled
    assert np.all(y[:5] == 0)
    # clipping bounds the magnitude except for the injected outliers
    mag = np.hypot(y[:, 0], y[:, 1])
    finite = np.isfinite(mag)
    assert np.isnan(mag[10])
    outliers = mag[finite] > 10
    assert outliers.sum() == 3
    assert np.all(mag[finite][~outliers] <= 0.6 + 1e-5)


def test_write_dataset_records_fixture_metadata(tmp_path):
    write_dataset(tmp_path, n=1000, seed=3, fs=800e6, bandwidth=200e6,
                  impairments=Impairments(delay_samples=2), fmt="csv")
    meta = json.loads((tmp_path / "fixture.json").read_text())
    assert meta["synthetic"] is True
    assert meta["impairments"]["delay_samples"] == 2
    header = (tmp_path / "data.csv").read_text().splitlines()[0]
    assert header == "I_in,Q_in,I_out,Q_out"
