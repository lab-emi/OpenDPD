"""Executable checks of the train/validation/test boundary and polynomial lineage."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opendpd.core import arena, arena_engine as engine, arena_runner as runner
from tests.unit.test_arena_scoring import calibrated, cases, score  # noqa: F401


@pytest.mark.parametrize("identifier", [board["board_id"] for board in arena.BOARDS])
def test_training_loader_never_opens_test_arrays(identifier, monkeypatch):
    record = arena.calibration()[identifier]
    actual_load, accessed = np.load, []

    class TrainingArchive:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key in {"x_train", "y_train", "x_val", "y_val"}, "held-out array was opened"
            accessed.append(key)
            return self.archive[key]

    def guarded_load(path, *args, **kwargs):
        value = actual_load(path, *args, **kwargs)
        return TrainingArchive(value) if Path(path).name == record["data_file"] else value

    monkeypatch.setattr(runner.np, "load", guarded_load)
    condition = runner.TrainingCondition(identifier, "cpu")
    assert accessed == ["x_train", "y_train", "x_val", "y_val"]
    assert set(condition.data) == set(accessed)
    assert not hasattr(condition, "metrics") and not hasattr(condition, "baseline_metrics")
    assert not condition.teacher.training and not any(p.requires_grad for p in condition.teacher.parameters())
    with actual_load(arena.ASSETS / record["data_file"], allow_pickle=False) as original:
        for name, value in condition.data.items():
            np.testing.assert_array_equal(value, original[name])
    assert condition.peak == record["peak_limit"] and condition.gain == record["reference_gain"]


class TrainingOnly(dict):
    def __getitem__(self, key):
        assert key in {"x_train", "y_train"}, "polynomial fitting accessed a held-out split"
        return super().__getitem__(key)


def polynomial_condition(length=700):
    x = np.random.default_rng(8).normal(0, .1, (length, 2)).astype(np.float32)

    class PA(torch.nn.Module):
        def forward(self, value):
            return value * 1.5

    return SimpleNamespace(identifier="train-only", device="cpu", teacher=PA(), gain=1.5, peak=2.,
                           data=TrainingOnly(x_train=x, y_train=np.full_like(x, 123.)))


@pytest.mark.parametrize("key,parameters", [
    ("mp_ls", dict(K=3, Q=2, rcond=1e-4)),
    ("gmp_ls", dict(Ka=3, La=2, Kb=1, Lb=2, Mb=1, Kc=1, Lc=2, Mc=1, rcond=1e-4)),
])
def test_mp_gmp_use_training_pa_feedback_and_never_ilc(tmp_path, monkeypatch, key, parameters):
    from opendpd.core import ilc
    monkeypatch.setattr(ilc, "learn", lambda *args, **kwargs: pytest.fail("MP/GMP called ILC"))
    condition = polynomial_condition()
    seen = {}
    make_basis, least_squares = engine.basis, engine.fit_least_squares

    def basis(backbone, params, post_input):
        seen["post_input"] = post_input.copy()
        return make_basis(backbone, params, post_input)

    def fit(phi, target, rcond):
        seen["target"] = target.copy()
        return least_squares(phi, target, rcond)

    monkeypatch.setattr(engine, "basis", basis)
    monkeypatch.setattr(engine, "fit_least_squares", fit)
    _, info = engine.fit_polynomial(condition, key, parameters, tmp_path, lambda *args: None)
    x = condition.data["x_train"]
    np.testing.assert_array_equal(seen["post_input"], engine.to_complex(1.5 * x) / condition.gain)
    np.testing.assert_array_equal(seen["target"], engine.to_complex(x))
    assert info["fit"]["n_observations"] == len(x)
    assert "ilc_iterations" not in info


def test_excluded_historical_ilc_path_uses_only_training_prefix(tmp_path, monkeypatch):
    from opendpd.core import ilc
    condition = polynomial_condition(17000)
    seen = []

    def learn(plant, reference, **kwargs):
        seen.append(reference.copy())
        np.testing.assert_array_equal(reference, engine.to_complex(condition.data["x_train"][:16384]))
        assert kwargs["iterations"] == 30
        return SimpleNamespace(input=.9 * reference, history=[{}, {}])

    monkeypatch.setattr(ilc, "learn", learn)
    _, info = engine.fit_polynomial(condition, "ilc_dpd", dict(K=2, Q=2, rcond=1e-6),
                                    tmp_path, lambda *args: None)
    assert len(seen) == 1 and info["fit"]["n_observations"] == 16384 and info["test_feedback"] is False
    assert "ilc_dpd" not in {entry.key for entry in arena.bundled_backbones()}
    with pytest.raises(ValueError, match="excluded"):
        arena.sweep("ilc_dpd")
    with pytest.raises(ValueError, match="excluded"):
        runner.evaluate_request(dict(protocol_sha256=arena.protocol().protocol_sha256,
                                     board_id="apa-200mhz-b", backbone="ilc_dpd"), tmp_path / "result.json")


def test_validation_diagnostics_cannot_change_final_test_fom(calibrated):  # noqa: F811
    evidence = cases(calibrated)
    before = score(evidence)
    for case in evidence:
        case.update(selected_validation_nmse_db=999., selected_validation_feasible=False,
                    validation_evm_db=-999., validation_aclr_db=-999.)
    assert score(evidence) == before
    for case in evidence:
        case["judges"][0]["evm_db"] -= 2.
    assert score(evidence)["score"] == pytest.approx(before["score"] + 1.)


@pytest.mark.parametrize("split", [None, "train", "val"])
def test_scoring_refuses_non_test_observations(calibrated, split):  # noqa: F811
    evidence = cases(calibrated)
    evidence[0]["evaluation_split"] = split
    with pytest.raises(ValueError, match="test observations"):
        score(evidence)
