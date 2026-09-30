"""Actual CUDA replay matches eager Arena DPD updates through a frozen PA."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from models import CoreModel
from opendpd.core import arena
from opendpd.core.arena_cuda import ARENA_REPLAY_BACKBONES, make_arena_step

NEW_BACKBONES = ["vdlstm", "tcn", "mcldnn", "pgjanet", "dvrjanet", "bojanet", "apnrru", "gmp"]
# The parameter sweep also trains far narrower networks than the reviewed width of 8:
# the smallest registered configuration of every replayed backbone (BoJANET: one hidden unit).
SMALLEST_SWEEP_POINTS = sorted((key, next(point["budget"] for point in arena.sweep(key)
                                          if point["model_parameters"] is not None))     # GMP's configuration is {}
                               for key in ARENA_REPLAY_BACKBONES)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("zero_iq", [False, True])
def test_gmp_direct_padding_matches_legacy_host_padding(device, zero_iq):
    from backbones.gmp import GMP
    torch.manual_seed(319)
    actual = GMP(memory_length=3, degree=3).to(device)
    actual.reset_parameters()
    legacy = copy.deepcopy(actual)
    x = torch.randn(2, 19, 2, device=device) * 0.1
    x[:, ::3] = 0
    if zero_iq:
        x.zero_()
    inputs = [x.clone().requires_grad_(), x.clone().requires_grad_()]
    zeros = torch.zeros

    def legacy_padding(*args, **kwargs):
        # Reproduce only the previous padding allocation while retaining the
        # actual GMP forward math. The output allocation is unchanged.
        if args and args[0] == (2, 2):
            target = kwargs.pop("device")
            return zeros(*args, **kwargs).to(target)
        return zeros(*args, **kwargs)

    observed = actual(inputs[0], None)
    with patch("backbones.gmp.torch.zeros", side_effect=legacy_padding):
        expected = legacy(inputs[1], None)
    target = torch.randn_like(observed)
    nn.functional.mse_loss(observed, target).backward()
    nn.functional.mse_loss(expected, target).backward()
    for left, right in ((observed, expected), (inputs[0].grad, inputs[1].grad),
                        (actual.Weight.grad, legacy.Weight.grad)):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def replay_matches_eager(build_dpd):
    from opendpd.core.arena_engine import ArenaCascade
    torch.manual_seed(317)
    original = ArenaCascade(build_dpd(), CoreModel(2, 8, 1, "tres_gru"), peak=0.2).cuda().train()
    original.freeze_pa_model()
    accelerated = copy.deepcopy(original)
    frozen = {name: parameter.detach().clone() for name, parameter in original.named_parameters()
              if not parameter.requires_grad}
    parameters = [tuple(parameter for parameter in net.parameters() if parameter.requires_grad)
                  for net in (original, accelerated)]
    optimizers = [torch.optim.AdamW(items, lr=0.005) for items in parameters]
    criterion = nn.MSELoss()
    fast = make_arena_step(accelerated, criterion, parameters[1], 0.2)
    assert fast is not None
    # Include exact zero IQ in regular frames, then a short batch that must fall
    # back to eager without discarding the captured gradient buffers.
    x = torch.randn(4, 200, 2, device="cuda") * 0.1
    x[:, ::7] = 0
    y = x[:, 50:150] * 1.2
    for epoch in range(3):
        for optimizer in optimizers:
            optimizer.param_groups[0]["lr"] = 0.005 * 0.5 ** epoch
        for batch in ((x, y), (torch.zeros_like(x), torch.zeros_like(y)), (x[:3], y[:3]), (x, y)):
            losses = []
            gradients = []
            for index, (net, optimizer) in enumerate(zip((original, accelerated), optimizers)):
                rng = torch.cuda.get_rng_state().clone()
                loss = fast(*batch) if index else None
                if index and batch[0].shape[0] != x.shape[0]:
                    assert loss is None
                if loss is None:
                    optimizer.zero_grad(set_to_none=False)
                    loss = criterion(net(batch[0]), batch[1])
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters[index], 0.2)
                gradients.append([parameter.grad.detach().clone() for parameter in parameters[index]])
                optimizer.step()
                losses.append(loss.detach())
                assert torch.equal(rng, torch.cuda.get_rng_state())
            torch.testing.assert_close(losses[0], losses[1], rtol=2e-5, atol=2e-7)
            for eager_gradient, replay_gradient in zip(*gradients):
                torch.testing.assert_close(eager_gradient, replay_gradient, rtol=3e-5, atol=3e-6)
        for left, right in zip(original.parameters(), accelerated.parameters()):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
        for eager_state, replay_state in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for name in eager_state:
                torch.testing.assert_close(eager_state[name], replay_state[name], rtol=3e-5, atol=3e-6)
    assert fast.graph is not None and not fast.failed
    for name, value in frozen.items():
        assert torch.equal(dict(accelerated.named_parameters())[name], value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone", NEW_BACKBONES)
def test_arena_replay_preserves_frozen_pa_optimizer_rng_and_lr(backbone):
    replay_matches_eager(lambda: CoreModel(2, 8, 1, backbone, window_size=4, num_dvr_units=3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone,budget", SMALLEST_SWEEP_POINTS)
def test_arena_replay_holds_for_the_smallest_registered_sweep_configuration(backbone, budget):
    from opendpd.core.arena_engine import build_model
    replay_matches_eager(lambda: build_model(backbone, arena.model_parameters(backbone, budget)))


def test_only_the_engine_cascade_is_reviewed_for_replay():
    from opendpd.core import arena_cuda, arena_engine, arena_runner

    def cascade(kind):
        return kind(CoreModel(2, 8, 1, "gru"), CoreModel(2, 8, 1, "tres_gru"), peak=0.2)

    class ArenaCascade(arena_engine.ArenaCascade):        # The reviewed name, defined elsewhere.
        pass

    assert arena_runner.ArenaCascade is arena_engine.ArenaCascade
    assert arena_engine.ArenaCascade.__module__ == "opendpd.core.arena_engine"
    assert arena_cuda._supported_model(cascade(arena_engine.ArenaCascade))
    assert not arena_cuda._supported_model(cascade(ArenaCascade))
    trainable_pa = cascade(arena_engine.ArenaCascade)
    next(trainable_pa.pa_model.parameters()).requires_grad = True
    assert not arena_cuda._supported_model(trainable_pa)
    # Every replayed backbone is a member of the sweep; its registered sizes are what Arena trains.
    assert {key for key, _ in SMALLEST_SWEEP_POINTS} == set(ARENA_REPLAY_BACKBONES)
    assert dict(SMALLEST_SWEEP_POINTS)["mcldnn"] == 1000 and dict(SMALLEST_SWEEP_POINTS)["apnrru"] == 500
    assert arena.model_parameters("bojanet", 250)["hidden_size"] == 1


def test_arena_cpu_and_unreviewed_models_fall_back():
    for net in (CoreModel(2, 8, 1, "tcn"), nn.Linear(2, 2), CoreModel(2, 8, 1, "qgru")):
        assert make_arena_step(net, nn.MSELoss(), tuple(net.parameters()), 0.2) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arena_replay_rejects_mutable_buffers_hooks_and_stochastic_modules(monkeypatch):
    def fresh():
        return CoreModel(2, 8, 1, "tcn").cuda()

    for mutation in (
            lambda net: net.register_buffer("observer_count", torch.zeros(1, device="cuda")),
            lambda net: net.register_forward_hook(lambda *args: None),
            lambda net: net.add_module("dropout", nn.Dropout2d(0.5)),
            lambda net: setattr(net, "statistics", {}),
            lambda net: setattr(net, "activation_post_process", nn.Identity())):
        net = fresh()
        mutation(net)
        assert make_arena_step(net, nn.MSELoss(), tuple(net.parameters()), 0.2) is None
    net = fresh()
    assert make_arena_step(net, nn.MSELoss(), tuple(reversed(tuple(net.parameters()))), 0.2) is None
    monkeypatch.setenv("OPENDPD_DISABLE_CUDA_FAST_PATH", "1")
    assert make_arena_step(net, nn.MSELoss(), tuple(net.parameters()), 0.2) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arena_capture_failure_returns_eager_fallback(monkeypatch):
    net = CoreModel(2, 8, 1, "tcn").cuda().train()
    fast = make_arena_step(net, nn.MSELoss(), tuple(net.parameters()), 0.2)

    def fail(*args):
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr(fast, "capture", fail)
    batch = torch.randn(2, 32, 2, device="cuda")
    assert fast(batch, batch) is None
    assert fast.failed and fast.graph is None
    assert fast(batch, batch) is None
