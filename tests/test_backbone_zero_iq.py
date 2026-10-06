"""Muted IQ and evaluation padding must not poison phase-normalized models."""

import importlib

import pytest
import torch

from backbones.finite_iq import amplitude, phase, phase_denominator
from models import CascadedModel, CoreModel

PHASE_FEATURE_MODELS = ["dgru", "tcn", "vdlstm", "deltagru", "deltajanet", "apnrru",
                       "tres_gru", "tres_deltagru", "bojanet", "pgjanet", "dvrjanet",
                       "rvtdcnn", "mcldnn", "qgru_amp1"]
ALL_MODELS = PHASE_FEATURE_MODELS + ["gru", "lstm", "qgru", "gmp", "user_template"]


@pytest.mark.parametrize("key", ALL_MODELS)
@pytest.mark.parametrize("muted", [True, False])
def test_zero_iq_has_finite_outputs_and_gradients(key, muted):
    torch.manual_seed(4)
    net = CoreModel(2, 8, 1, key, num_dvr_units=3)
    iq = torch.randn(2, 32, 2) * 0.2
    iq[:, :16] = 0  # A leading mute and a complete zero-padded memory window.
    if muted:
        iq.zero_()
    iq.requires_grad_()
    output = net(iq)
    assert output.shape == iq.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(iq.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("key", ALL_MODELS)
def test_zero_iq_backpropagates_through_frozen_tres_teacher(key):
    torch.manual_seed(317)
    net = CascadedModel(CoreModel(2, 8, 1, key, num_dvr_units=3),
                        CoreModel(2, 8, 1, "tres_gru"))
    net.freeze_pa_model()
    iq = torch.randn(2, 32, 2) * .1
    iq[:, ::7] = 0
    output = net(iq)
    assert torch.isfinite(output).all()
    (output - iq * 1.2).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.dpd_model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in net.pa_model.parameters())


@pytest.mark.parametrize("key", PHASE_FEATURE_MODELS)
def test_nonzero_model_outputs_and_weight_gradients_are_bitwise_unchanged(key, monkeypatch):
    torch.manual_seed(31)
    net = CoreModel(2, 8, 1, key, num_dvr_units=3)
    iq = (torch.randn(2, 32, 2) * .2).requires_grad_()

    def run():
        net.zero_grad(set_to_none=True)
        iq.grad = None
        output = net(iq)
        output.square().mean().backward()
        gradients = [iq.grad.clone()] + [p.grad.clone() for p in net.parameters() if p.grad is not None]
        return output.detach(), gradients

    actual, actual_gradients = run()
    module = importlib.import_module("backbones." + key)
    monkeypatch.setattr(module, "amplitude", torch.sqrt)
    if hasattr(module, "phase_denominator"):
        monkeypatch.setattr(module, "phase_denominator", lambda magnitude: magnitude)
    if hasattr(module, "phase"):
        monkeypatch.setattr(module, "phase", lambda i, q: torch.atan2(q, i))
    expected, expected_gradients = run()
    assert torch.equal(actual, expected)
    assert len(actual_gradients) == len(expected_gradients)
    assert all(torch.equal(a, b) for a, b in zip(actual_gradients[1:], expected_gradients[1:]))
    # Guarding the shared phase denominator changes the order in which its
    # input-gradient branches accumulate; values may differ by float32 rounding.
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=1e-6, atol=1e-7)


def test_nonzero_amplitude_and_phase_are_unchanged():
    torch.manual_seed(3)
    iq = torch.randn(1024, 2, dtype=torch.float64)
    squared = iq.square().sum(-1).requires_grad_()
    expected = torch.sqrt(squared)
    actual = amplitude(squared)
    assert torch.equal(actual, expected)
    assert torch.equal(iq / phase_denominator(actual)[:, None], iq / expected[:, None])
    actual_grad = torch.autograd.grad(actual.sum(), squared, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), squared)[0]
    assert torch.equal(actual_grad, expected_grad)

    i, q = iq[:, 0].clone().requires_grad_(), iq[:, 1].clone().requires_grad_()
    expected_phase, actual_phase = torch.atan2(q, i), phase(i, q)
    assert torch.equal(expected_phase, actual_phase)
    expected_grad = torch.autograd.grad(expected_phase.sum(), (i, q), retain_graph=True)
    actual_grad = torch.autograd.grad(actual_phase.sum(), (i, q))
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))


def test_invalid_input_is_not_hidden_by_zero_handling():
    values = amplitude(torch.tensor([0., float("nan"), float("inf")]))
    assert values[0] == 0
    assert torch.isnan(values[1])
    assert torch.isinf(values[2])
